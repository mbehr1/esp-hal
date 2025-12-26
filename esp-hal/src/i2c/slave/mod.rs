#![cfg_attr(docsrs, procmacros::doc_replace)]
//! # Inter-Integrated Circuit (I2C) - Slave mode
//!
//! ## Overview
//!
//! This driver implements the I2C Slave mode. In this mode, the MCU responds to
//! and communicates with one or more master devices. The MCU acts as a slave
//! device on the I2C bus, identified by its unique I2C address.
//!
//! for register spec see e.g. here: https://docs.rs/esp32c6/0.21.0/esp32c6/i2c0/ctr/index.html
//! and in detail: https://documentation.espressif.com/esp32-c6_technical_reference_manual_en.pdf#i2c
//!
//! build e.g. `via cargo +nightly build --release --features="esp32c6" --target=riscv32imac-unknown-none-elf`

// TODOs:
// [] - impl read/write. Currently only on_event callback can handle data transfers
// [] - impl for non stretch i2c interface chips? (At least dont build...)
// [] - impl NAK if no data to write was provided
// [] - 2.9.14. check There are two ways to start the I2C controller in slave mode:
//      • Set I2C_SLV_TX_AUTO_START_EN, and the slave starts automatic transfer upon an address match;
//      • Clear I2C_SLV_TX_AUTO_START_EN, and always set I2C_TRANS_START before accepting any transfer.
//
// [] - check FIFO wrap logic PTR_EN...
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, rwlock::RwLock};
use enumset::{EnumSet, EnumSetType};

use crate::{
    Async, Blocking, DriverMode, PhantomData, any_peripheral,
    asynch::AtomicWaker,
    gpio::{
        DriveMode, InputSignal, OutputConfig, OutputSignal, PinGuard, Pull,
        interconnect::{self, PeripheralOutput},
    },
    handler,
    i2c::master::I2cAddress,
    interrupt::{self, InterruptHandler},
    pac::i2c0::RegisterBlock,
    private, ram,
    system::PeripheralGuard,
};
use portable_atomic::{AtomicUsize, Ordering};

/// I2C-specific transmission errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub enum Error {
    /// A timeout occurred during transmission.
    Timeout,
    /// The arbitration for the bus was lost.
    ArbitrationLost,
    /// The execution of the I2C command was incomplete.
    ExecutionIncomplete,
    /// Zero length read or write operation.
    ZeroLengthInvalid,
    /// The given address is invalid.
    AddressInvalid(I2cAddress),
}

impl embedded_hal::i2c::Error for Error {
    fn kind(&self) -> embedded_hal::i2c::ErrorKind {
        use embedded_hal::i2c::ErrorKind;

        match self {
            Self::ArbitrationLost => ErrorKind::ArbitrationLoss,
            _ => ErrorKind::Other,
        }
    }
}

/// I2C driver configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, procmacros::BuilderLite)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub struct Config {
    /// The I2C slave address.
    ///
    /// Supports both 7-bit (0x00..=0x7F) and 10-bit (0x000..=0x3FF) addresses.
    /// Use `I2cAddress::SevenBit(addr)` or `I2cAddress::TenBit(addr)`, or simply
    /// convert from `u8` for 7-bit or `u16` for automatic detection.
    ///
    /// Default value: 7-bit address 0x55.
    address: I2cAddress,

    /// rx_fifo_wm_threshold: The water mark threshold of rx fifo
    /// When reached, the rx fifo watermark interrupt is triggered and a SlaveEvent is sent to the user callback.
    ///
    /// Set to None to disable the watermark interrupt.
    /// Default value: half of fifo size
    rx_fifo_wm_threshold: Option<u8>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            address: I2cAddress::SevenBit(0x55),
            rx_fifo_wm_threshold: Some((FIFO_SIZE / 2) as u8),
        }
    }
}

/// I2C-specific configuration errors
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub enum ConfigError {
    /// Provided address is not valid.
    AddressInvalid,
}

impl core::error::Error for ConfigError {}

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConfigError::AddressInvalid => write!(f, "Provided address is invalid"),
        }
    }
}

#[procmacros::doc_replace]
/// I2C slave driver
///
/// ## Example
///
/// ```rust, no_run
/// # {before_snippet}
/// use esp_hal::i2c::slave::{Config, I2c};
/// let mut i2c = I2c::new(peripherals.I2C0, Config::default())?
///     .with_sda(peripherals.GPIO1)
///     .with_scl(peripherals.GPIO2);
///
/// let mut data = [0u8; 128];
/// let bytes_read = i2c.read(&mut data)?;
/// # {after_snippet}
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct I2c<'d, Dm: DriverMode> {
    i2c: AnyI2c<'d>,
    phantom: PhantomData<Dm>,
    guard: PeripheralGuard,
    config: DriverConfig,
}

#[derive(Debug)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
struct DriverConfig {
    config: Config,
    sda_pin: PinGuard,
    scl_pin: PinGuard,
}

#[instability::unstable]
impl<Dm: DriverMode> embassy_embedded_hal::SetConfig for I2c<'_, Dm> {
    type Config = Config;
    type ConfigError = ConfigError;

    fn set_config(&mut self, config: &Self::Config) -> Result<(), Self::ConfigError> {
        self.apply_config(config)
    }
}

impl<Dm: DriverMode> embedded_hal::i2c::ErrorType for I2c<'_, Dm> {
    type Error = Error;
}

impl<'d> I2c<'d, Blocking> {
    #[procmacros::doc_replace]
    /// Create a new I2C slave instance.
    ///
    /// ## Errors
    ///
    /// A [`ConfigError`] variant will be returned if the slave address
    /// passed in config is invalid.
    ///
    /// ## Example
    ///
    /// ```rust, no_run
    /// # {before_snippet}
    /// use esp_hal::i2c::slave::{Config, I2c};
    /// let i2c = I2c::new(peripherals.I2C0, Config::default())?
    ///     .with_sda(peripherals.GPIO1)
    ///     .with_scl(peripherals.GPIO2);
    /// # {after_snippet}
    /// ```
    pub fn new(i2c: impl Instance + 'd, config: Config) -> Result<Self, ConfigError> {
        let guard = PeripheralGuard::new(i2c.info().peripheral);

        let sda_pin = PinGuard::new_unconnected();
        let scl_pin = PinGuard::new_unconnected();

        let mut i2c = I2c {
            i2c: i2c.degrade(),
            phantom: PhantomData,
            guard,
            config: DriverConfig {
                config,
                sda_pin,
                scl_pin,
            },
        };
        // we need the async handler even in blocking mode to handle interrupts
        i2c.set_interrupt_handler(i2c.driver().info.async_handler);

        i2c.apply_config(&config)?;

        Ok(i2c)
    }

    /// Reconfigures the driver to operate in [`Async`] mode.
    pub fn into_async(mut self) -> I2c<'d, Async> {
        self.set_interrupt_handler(self.driver().info.async_handler);

        I2c {
            i2c: self.i2c,
            phantom: PhantomData,
            guard: self.guard,
            config: self.config,
        }
    }

    #[instability::unstable]
    pub fn set_interrupt_handler(&mut self, handler: InterruptHandler) {
        self.i2c.set_interrupt_handler(handler);
    }
    /// Listen for the given interrupts
    #[instability::unstable]
    pub fn listen(&mut self, interrupts: impl Into<EnumSet<Event>>) {
        self.i2c.info().enable_listen(interrupts.into(), true)
    }

    /// Unlisten the given interrupts
    #[instability::unstable]
    pub fn unlisten(&mut self, interrupts: impl Into<EnumSet<Event>>) {
        self.i2c.info().enable_listen(interrupts.into(), false)
    }

    /// Gets asserted interrupts
    #[instability::unstable]
    pub fn interrupts(&mut self) -> EnumSet<Event> {
        self.i2c.info().interrupts()
    }

    /// Resets asserted interrupts
    #[instability::unstable]
    pub fn clear_interrupts(&mut self, interrupts: EnumSet<Event>) {
        self.i2c.info().clear_interrupts(interrupts)
    }

    /// Register callback for I2C slave events
    ///
    /// This is currently the only way to handle data transfers in slave mode.
    pub fn register_callbacks(&mut self, on_event: I2cSlaveEventCallback) {
        // try in a loop to get the lock
        loop {
            let callback_lock = self.i2c.state().callback.try_write();
            if callback_lock.is_err() {
                continue;
            }
            let mut callback_lock = callback_lock.unwrap();
            *callback_lock = Some(on_event);
            break;
        }
        // now start listening to interrupts
        // unlisten to all:
        self.unlisten(EnumSet::all());

        // enable the ones we do need:
        let mut ints_to_enable = Event::TransComplete | Event::SlaveStretch | Event::Nack;
        if self.config.config.rx_fifo_wm_threshold.is_some() {
            ints_to_enable |= Event::RxFifoWatermark;
        }
        self.listen(ints_to_enable);
    }
}

impl private::Sealed for I2c<'_, Blocking> {}

#[derive(Debug, EnumSetType)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
#[instability::unstable]
pub enum Event {
    /// Triggered when op_code of the master indicates an END command and an END
    /// condition is detected.
    EndDetect,

    /// Triggered when the I2C controller detects a STOP bit.
    TransComplete,

    /// Triggered when the RX FIFO reaches the configured watermark level if configured.
    RxFifoWatermark,

    /// Main interrupt for slave stretch handling
    SlaveStretch,

    /// Triggered when a NACK is received from the master usually at end of read transfer
    Nack,
}

#[instability::unstable]
impl crate::interrupt::InterruptConfigurable for I2c<'_, Blocking> {
    fn set_interrupt_handler(&mut self, handler: InterruptHandler) {
        self.i2c.set_interrupt_handler(handler);
    }
}

impl<'d, Dm> I2c<'d, Dm>
where
    Dm: DriverMode,
{
    fn driver(&self) -> Driver<'_> {
        Driver {
            info: self.i2c.info(),
            state: self.i2c.state(),
            config: &self.config,
        }
    }

    /// Connect a pin to the I2C SDA signal.
    ///
    /// This will replace previous pin assignments for this signal.
    pub fn with_sda(mut self, sda: impl PeripheralOutput<'d>) -> Self {
        let info = self.driver().info;
        let input = info.sda_input;
        let output = info.sda_output;
        Driver::connect_pin(sda.into(), input, output, &mut self.config.sda_pin);
        self
    }
    #[procmacros::doc_replace]
    /// Connect a pin to the I2C SCL signal.
    ///
    /// This will replace previous pin assignments for this signal.
    ///
    /// ## Example
    ///
    /// ```rust, no_run
    /// # {before_snippet}
    /// use esp_hal::i2c::slave::{Config, I2c};
    /// const DEVICE_ADDR: u8 = 0x77;
    /// let i2c = I2c::new(peripherals.I2C0, Config::default())?.with_scl(peripherals.GPIO2);
    /// # {after_snippet}
    /// ```
    pub fn with_scl(mut self, scl: impl PeripheralOutput<'d>) -> Self {
        let info = self.driver().info;
        let input = info.scl_input;
        let output = info.scl_output;
        Driver::connect_pin(scl.into(), input, output, &mut self.config.scl_pin);
        self
    }

    #[procmacros::doc_replace]
    /// Writes data to be sent to the master
    pub fn write(&mut self, buffer: &[u8]) -> Result<usize, Error> {
        if buffer.is_empty() {
            return Err(Error::ZeroLengthInvalid);
        }
        unimplemented!() // TODO! impl like in https://github.com/espressif/esp-idf/blob/487551888a46971a07e33caef3312fe3a6f5cf68/components/esp_driver_i2c/i2c_slave.c#L362
        // main task here:
        // fill fifo
        // i2c_ll_slave_enable_tx_it(hal->dev);
        // i2c_ll_slave_clear_stretch(hal->dev);
        // TODO currently only the on_event callback can write data to the fifo
    }

    #[procmacros::doc_replace]
    /// Applies a new configuration.
    ///
    /// ## Errors
    ///
    /// A [`ConfigError`] variant will be returned if bus frequency or timeout
    /// passed in config is invalid.
    ///
    /// ## Example
    ///
    /// ```rust, no_run
    /// # {before_snippet}
    /// use esp_hal::i2c::slave::{Config, I2c};
    /// let mut i2c = I2c::new(peripherals.I2C0, Config::default())?;
    ///
    /// i2c.apply_config(&Config::default())?;
    /// # {after_snippet}
    /// ```
    pub fn apply_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        self.config.config = *config;
        self.driver().setup(config)?;
        // TODO needed? check esp-idf/esp_driver_i2c  self.driver().reset_fsm(false);
        Ok(())
    }
}

/// size of the i2c tx and rx fifos
pub const FIFO_SIZE: usize = property!("i2c_master.fifo_size");

// MARK: async_handler
/// The asynchronous interrupt handler for the I2C slave driver.
/// needs to be similar to esp-idf i2c_slave_isr_handler
#[ram]
fn async_handler(info: &Info, state: &State) {
    let callback_lock = state.callback.try_read();
    if callback_lock.is_err() {
        // unable to get lock, panic or just return?
        panic!(
            "i2c slave async handler unable to get callbacks lock. Dont call register_callbacks after apply_config!"
        );
    }
    let callback_lock = callback_lock.unwrap();
    let on_event = match *callback_lock {
        Some(callbacks) => callbacks,
        None => {
            // Disable all interrupts. The I2C Future will check events based on the
            // interrupt status bits.
            info.regs().int_ena().write(|w| unsafe { w.bits(0) });
            state.waker.wake();
            return;
        }
    };
    // unlock here to avoid the CriticalSectionRawMutex being locked
    drop(callback_lock);
    let regs = info.regs();

    // i2c_ll_get_intr_mask(hal->dev, &int_mask);
    let ints = regs.int_st().read(); //  int_raw().read(); // TODO raw or masked onces only?

    // i2c_ll_get_rxfifo_cnt(hal->dev, &rx_fifo_exist_len);
    let rx_fifo_exist_len = regs.sr().read().rxfifo_cnt().bits();
    // let slave_is_write_by_master = regs.sr().read().slave_rw().bit_is_clear();
    if ints.nack().bit_is_set() {
        // handle nack by resetting fifo
        reset_tx_fifo(regs);
        // todo inform user about how many bytes were written?
    }
    if ints.trans_complete().bit_is_set() {
        // slave gets this on write transfer end from master without read
        let prev_transaction_rx_cnt = state.cur_transaction_rx_cnt.swap(0, Ordering::Relaxed);

        if rx_fifo_exist_len > 0 || prev_transaction_rx_cnt > 0 {
            // todo or call this even if 0/0 to inform about end of transfer?
            let mut data = [0u8; FIFO_SIZE];
            let did_read = read_rx_fifo(regs, &mut data);
            let _ = on_event(
                SlaveEvent::TransComplete(prev_transaction_rx_cnt),
                &data[0..did_read],
                &mut [],
            );
        }
    } else if ints.rxfifo_wm().bit_is_set() {
        // we check this in the else part only as the trans_complete would read the fifo anyhow already
        // and the Watermark reason might be misleading if it's the last data
        // pxHigherPriorityTaskWoken |= i2c_slave_handle_rx_fifo(i2c_slave, rx_fifo_exist_len);

        let mut data = [0u8; FIFO_SIZE];
        let did_read = read_rx_fifo(regs, &mut data);
        let prev_transaction_rx_cnt = state
            .cur_transaction_rx_cnt
            .fetch_add(did_read, Ordering::Relaxed);
        let _ = on_event(
            SlaveEvent::RxFifoWatermark(prev_transaction_rx_cnt),
            &data[0..did_read],
            &mut [],
        );
    }
    if ints.slave_stretch().bit_is_set() {
        let cause = regs.sr().read().stretch_cause().bits();
        match cause {
            0 => {
                // I2C_SLAVE_STRETCH_CAUSE_ADDRESS_MATCH
                // check if data in rx fifo, if so, process it
                // we treat this as an end of the transaction, here from W ... data ... R
                let prev_transaction_rx_cnt =
                    state.cur_transaction_rx_cnt.swap(0, Ordering::Relaxed);
                let mut recvd_data = [0u8; FIFO_SIZE];
                let did_read = read_rx_fifo(regs, &mut recvd_data);

                let mut data_to_write = [0u8; FIFO_SIZE];
                let tx_fifo_exist_len = regs.sr().read().txfifo_cnt().bits();
                //TODO this fails sometimes! Analyse assert_eq!(tx_fifo_exist_len, 0); // should be 0 at address match
                let space_in_fifo = FIFO_SIZE.saturating_sub(tx_fifo_exist_len as usize);
                let to_write = on_event(
                    SlaveEvent::StretchAddrMatch(prev_transaction_rx_cnt),
                    &recvd_data[0..did_read],
                    &mut data_to_write[0..space_in_fifo],
                );
                // TODO how can we end/nack the transaction here if no data to write?
                // todo int for stretch should be cleared before the stretch is cleared! (according to e.g. 29.6.8.2 examples)
                slave_handle_write(regs, &data_to_write[0..to_write]); // does stretch clear as well
            }
            1 => {
                // I2C_SLAVE_STRETCH_CAUSE_TX_EMPTY
                let mut data = [0u8; FIFO_SIZE];
                let tx_fifo_exist_len = regs.sr().read().txfifo_cnt().bits() as usize;
                assert_eq!(tx_fifo_exist_len, 0); // should be 0 at stretch cause tx empty
                let to_write = on_event(SlaveEvent::StretchTxEmpty, &[], &mut data);
                slave_handle_write(regs, &data[0..to_write]); // does stretch clear as well
                // TODO how can we end/nack the transaction here if no data to write?
            }
            2 => {
                // I2C_SLAVE_STRETCH_CAUSE_RX_FULL
                // we handled rx fifo probably already above, so lets check if there is still data:
                let mut data = [0u8; FIFO_SIZE];
                let did_read = read_rx_fifo(regs, &mut data);
                if did_read > 0 {
                    let prev_transaction_rx_cnt = state
                        .cur_transaction_rx_cnt
                        .fetch_add(did_read, Ordering::Relaxed);
                    let _ = on_event(
                        SlaveEvent::StretchRxFull(prev_transaction_rx_cnt),
                        &data[0..did_read],
                        &mut [],
                    );
                }
                // in any case clear the stretch
                regs.scl_stretch_conf()
                    .modify(|_, w| w.slave_scl_stretch_clr().set_bit());
            }
            3 => { // I2C_SLAVE_STRETCH_CAUSE_SENDING_ACK
                // only needed if we do nack/ack manually and e.g. want to say: no more data to read
            }
            _ => {
                // unknown cause, ignore/panic?
            }
        }
    }
    if ints.end_detect().bit_is_set() {
        // TODO check if this is available for slave at all!
        // inform on_event callback how much was transferred? (by remaining fifo size? and clear fifo size?)
        state.cur_transaction_rx_cnt.store(0, Ordering::Relaxed);
    }
    // we're not using the txfifo_wm int as we handle it via the stretch int tx_empty

    // i2c_ll_clear_intr_mask(hal->dev, int_mask);
    regs.int_clr().write(|w| unsafe { w.bits(ints.bits()) });
    // TODO do this after processing? Does any of the handle_... function trigger a new interrupt?
}

// TODO: add a test on master side that sends in one transfer e.g. 99 bytes (>fifo_size)
/// From the spec: 29.4.10:
///  In FIFO mode, RX RAM of a slave may also wrap around to receive data larger than the FIFO depth. Set
///  I2C_FIFO_PRT_EN and clear I2C_RX_FULL_ACK_LEVEL. If data already received (to be overwritten) is larger
///  than I2C_RXFIFO_WM_THRHD (slave), an I2C_RXFIFO_WM_INT (slave) interrupt is generated. After receiving
///  the interrupt, software continues reading from I2C_DATA_REG (slave).

/// Read all data from rx fifo into buffer
///
/// If the buffer is smaller than the fifo content, only part of the data is read.
/// Returns number of bytes read
#[ram]
fn read_rx_fifo(regs: &RegisterBlock, buffer: &mut [u8]) -> usize {
    let rx_fifo_exist_len = regs.sr().read().rxfifo_cnt().bits();
    let to_read = core::cmp::min(rx_fifo_exist_len as usize, buffer.len());
    for i in 0..to_read {
        buffer[i] = regs.data().read().fifo_rdata().bits(); // as u8;
    }
    to_read
}

/// handle slave write
#[ram]
fn slave_handle_write(regs: &RegisterBlock, data: &[u8]) {
    let free_size = get_tx_fifo_free_size(regs) as usize;
    let to_write = core::cmp::min(free_size, data.len());
    for i in 0..to_write {
        regs.data()
            .write(|w| unsafe { w.fifo_rdata().bits(data[i]) }); // same register for read and write???
    }
    // i2c_ll_slave_enable_tx_it(hal->dev);
    // we dont use the tx_wm int

    // i2c_ll_slave_clear_stretch(hal->dev);
    regs.scl_stretch_conf()
        .modify(|_, w| w.slave_scl_stretch_clr().set_bit());

    // todo if no data to write stop the transaction by releasing the stretch or actively sending NACK?
    // TODO see 29.4.10 TX/RX RAM Data Storage on how to e.g. write data to tx fifo start... (direct access)
}

/// get free tx fifo size
///
/// static inline void i2c_ll_get_txfifo_len(i2c_dev_t *hw, uint32_t *length)
///    *length = (hw->sr.txfifo_cnt >= SOC_I2C_FIFO_LEN) ? 0 : (SOC_I2C_FIFO_LEN - hw->sr.txfifo_cnt);
#[ram]
fn get_tx_fifo_free_size(regs: &RegisterBlock) -> u8 {
    let txfifo_cnt = regs.sr().read().txfifo_cnt().bits();
    if txfifo_cnt >= FIFO_SIZE as u8 {
        0
    } else {
        (FIFO_SIZE as u8) - txfifo_cnt
    }
}

/// reset tx fifo:
///
/// static inline void i2c_ll_txfifo_rst(i2c_dev_t *hw)
// {
//     hw->fifo_conf.tx_fifo_rst = 1;
//     hw->fifo_conf.tx_fifo_rst = 0;
// }
#[ram]
fn reset_tx_fifo(regs: &RegisterBlock) {
    regs.fifo_conf().modify(|_, w| w.tx_fifo_rst().set_bit());
    regs.fifo_conf().modify(|_, w| w.tx_fifo_rst().clear_bit());
    //     i2c_ll_master_fsm_rst(i2c_slave->base->hal.dev);
    regs.ctr().modify(|_, w| w.fsm_rst().set_bit());
}

// MARK: Driver
#[allow(dead_code)] // Some versions don't need `state`
#[derive(Clone, Copy)]
struct Driver<'a> {
    info: &'a Info,
    state: &'a State,
    config: &'a DriverConfig,
}

impl Driver<'_> {
    fn regs(&self) -> &RegisterBlock {
        self.info.regs()
    }
    fn connect_pin(
        pin: crate::gpio::interconnect::OutputSignal<'_>,
        input: InputSignal,
        output: OutputSignal,
        guard: &mut PinGuard,
    ) {
        // avoid the pin going low during configuration
        pin.set_output_high(true);

        pin.apply_output_config(
            &OutputConfig::default()
                .with_drive_mode(DriveMode::OpenDrain)
                .with_pull(Pull::Up),
        );
        pin.set_output_enable(true);
        pin.set_input_enable(true);

        input.connect_to(&pin);

        *guard = interconnect::OutputSignal::connect_with_guard(pin, output);
    }

    // MARK: init_slave

    fn init_slave(&self) {
        // TODO from esp-idf i2c_new_slave_device: https://github.com/espressif/esp-idf/blob/487551888a46971a07e33caef3312fe3a6f5cf68/components/esp_driver_i2c/i2c_slave.c#L244
        self.regs().ctr().write(|w| {
            // from i2c_hal_slave_init:
            w.ms_mode().clear_bit(); // slave mode
            w.sda_force_out().set_bit(); // sda open drain
            w.scl_force_out().set_bit(); // scl open drain
            #[cfg(i2c_master_has_arbitration_en)]
            w.arbitration_en().clear_bit();
            // Use Most Significant Bit first for sending and receiving data
            w.tx_lsb_first().clear_bit();
            w.rx_lsb_first().clear_bit();
            w
        });
        // reset tx/rx fifos:
        self.reset_fifo();
        // end from i2c_hal_slave_init
        // i2c_ll_enable_fifo_mode(...true):
        self.regs().fifo_conf().modify(|_, w| {
            w.nonfifo_en().clear_bit(); // enable fifo mode
            w
        });
    }

    fn set_slave_addr(&self, address: I2cAddress) -> Result<(), ConfigError> {
        match address {
            I2cAddress::SevenBit(addr) if addr <= 0x7F => {
                // set slave_addr.addr_10bit_en to false/0
                // set slave_addr.slave_addr to addr
                self.regs().slave_addr().write(|w| unsafe {
                    w.slave_addr().bits(addr as u16);
                    w.addr_10bit_en().bit(false)
                });
                Ok(())
            }
            // anything else nyi
            _ => Err(ConfigError::AddressInvalid),
        }
    }

    /// Updates the configuration of the I2C peripheral.
    ///
    /// This function ensures that the configuration values, such as clock
    /// settings, SDA/SCL filtering, timeouts, and other operational
    /// parameters, which are configured in other functions, are properly
    /// propagated to the I2C hardware. This step is necessary to synchronize
    /// the software-configured settings with the peripheral's internal
    /// registers, ensuring that the hardware behaves according to the
    /// current configuration.
    fn update_registers(&self) {
        self.regs().ctr().modify(|_, w| w.conf_upgate().set_bit());
    }

    /// Configures the I2C peripheral with the specified frequency, clocks, and
    /// optional timeout.
    fn setup(&self, config: &Config) -> Result<(), ConfigError> {
        self.init_slave();

        // i2c_ll_set_slave_addr(hal->dev, slave_config->slave_addr, false):
        self.set_slave_addr(config.address)?;

        let regs = self.regs();
        // i2c_ll_set_tout(hal->dev, I2C_LL_MAX_TIMEOUT);
        // TODO: see 29.4.8 Timeout Control
        // 2 registers. I2C_TIME_OUT_EN and I2C_TIME_OUT_VALUE (<22...)
        regs.to().modify(|_, w| {
            unsafe {
                w.time_out_en().set_bit(); // todo? not done in i2c_slave.c?
                // lets use 100ms as timeout. Assuming 100kHz clock: 100ms / 10us = 10000
                // -> 2^14 = 16384 should be ok
                w.time_out_value().bits(14); // w.time_out_value().bits(0x0000001f); // Set a reasonable timeout value
            }
            w
        });

        /* TODO!
           I2C_CLOCK_SRC_ATOMIC() {
               i2c_ll_set_source_clk(hal->dev, slave_config->clk_source);
           }
           //i2c_ll_set_source_clk(i2c_dev_t *hw, i2c_clock_source_t src_clk):
           // src_clk : (1) for RTC_CLK, (0) for XTAL
           PCR.i2c_sclk_conf.i2c_sclk_sel = (src_clk == I2C_CLK_SRC_RC_FAST) ? 1 : 0;
        */
        regs.clk_conf().modify(|_, w| {
            w.sclk_sel().clear_bit()
            // w.sclk_div_num().bits((sclk_div - 1) as u8)
        });

        // now set slave_addr again? (now with 10bit support, above is set to false...)
        //  bool addr_10bit_en = slave_config->addr_bit_len != I2C_ADDR_BIT_LEN_7;
        //  i2c_ll_set_slave_addr(hal->dev, slave_config->slave_addr, addr_10bit_en);
        self.set_slave_addr(config.address)?;

        // i2c_ll_slave_broadcast_enable(hal->dev, slave_config->flags.broadcast_en);
        regs.ctr().modify(|_, w| {
            w.addr_broadcasting_en().clear_bit() // Disable broadcasting - respond only to our address
        });

        // i2c_ll_set_txfifo_empty_thr(hal->dev, SOC_I2C_FIFO_LEN / 2);
        regs.fifo_conf().modify(|_, w| {
            w.fifo_prt_en().set_bit();
            unsafe {
                w.txfifo_wm_thrhd().bits(0); // TODO does this disable the txfifo watermark? We dont want it for slave with stretch
            }
            w
        });
        // i2c_ll_set_rxfifo_full_thr(hal->dev, SOC_I2C_FIFO_LEN / 2);
        regs.fifo_conf().modify(|_, w| w.fifo_prt_en().set_bit());
        regs.ctr().write(|w| w.rx_full_ack_level().clear_bit());
        if let Some(rx_fifo_wm_threshold) = config.rx_fifo_wm_threshold {
            // TODO check max value?
            regs.fifo_conf()
                .modify(|_, w| unsafe { w.rxfifo_wm_thrhd().bits(rx_fifo_wm_threshold) });
        } else {
            regs.fifo_conf().modify(|_, w| unsafe {
                w.rxfifo_wm_thrhd()
                    // set to max value for this reg with 5 bits:
                    .bits(0x1F) // and later on we dont set the mask
            });
        }
        // i2c_ll_set_sda_timing(hal->dev, 10, 10);
        regs.sda_hold().write(|w| unsafe { w.time().bits(10u16) });
        regs.sda_sample().write(|w| unsafe { w.time().bits(10u16) });

        // i2c_ll_disable_intr_mask(hal->dev, I2C_LL_INTR_MASK);
        // #define I2C_LL_INTR_MASK          (0x3fff)
        let cur_ints = regs.int_ena().read().bits();
        regs.int_ena()
            .modify(|_, w| unsafe { w.bits(cur_ints & !0x3fff) });
        // i2c_ll_clear_intr_mask(hal->dev, I2C_LL_INTR_MASK);
        regs.int_clr().write(|w| unsafe { w.bits(0x3fff) });

        // i2c_ll_enable_intr_mask(hal->dev, I2C_LL_SLAVE_RX_EVENT_INTR);
        // #define I2C_LL_SLAVE_RX_EVENT_INTR  (I2C_TRANS_COMPLETE_INT_ENA_M | I2C_RXFIFO_WM_INT_ENA_M | I2C_SLAVE_STRETCH_INT_ENA_M)
        // enable is done when registering the callbacks

        // // Configure stretch
        // i2c_ll_slave_set_stretch_protect_num(hal->dev, I2C_LL_STRETCH_PROTECT_TIME);
        // #define I2C_LL_STRETCH_PROTECT_TIME  (0x3ff)
        // i2c_ll_slave_enable_scl_stretch(hal->dev, true);
        regs.scl_stretch_conf().modify(|_, w| unsafe {
            w.stretch_protect_num().bits(0x3ff);
            w.slave_scl_stretch_en().set_bit()
        });
        // i2c_ll_slave_clear_stretch(hal->dev);
        regs.scl_stretch_conf()
            .modify(|_, w| w.slave_scl_stretch_clr().set_bit());

        set_filter(self.regs(), Some(7), Some(7));

        // i2c_ll_update(hal->dev);
        self.update_registers();

        /* todo: from  https://github.com/espressif/esp-idf/blob/master/components/esp_driver_i2c/i2c_slave.c

        // Configure filter
        // FIXME if we ever change this we need to adapt `set_frequency` for ESP32
        set_filter(self.regs(), Some(7), Some(7));

        // Configure frequency
        self.set_frequency(config)?;

        self.update_registers();
        */
        Ok(())
    }

    /// resets the tx and rx fifo
    fn reset_fifo(&self) {
        reset_tx_fifo(self.regs());
        self.reset_rx_fifo();
    }

    /// resets the rx fifo
    fn reset_rx_fifo(&self) {
        self.regs().fifo_conf().modify(|_, w| {
            w.rx_fifo_rst().set_bit();
            w
        });
        self.regs().fifo_conf().modify(|_, w| {
            w.rx_fifo_rst().clear_bit();
            w
        });
    }
}

/// Sets the filter with a supplied threshold in clock cycles for which a
/// pulse must be present to pass the filter
fn set_filter(
    register_block: &RegisterBlock,
    sda_threshold: Option<u8>,
    scl_threshold: Option<u8>,
) {
    // register_block.sda_filter_cfg().modify(|_, w| {
    //     if let Some(threshold) = sda_threshold {
    //         unsafe { w.sda_filter_thres().bits(threshold) };
    //     }
    //     w.sda_filter_en().bit(sda_threshold.is_some())
    // });
    // register_block.scl_filter_cfg().modify(|_, w| {
    //     if let Some(threshold) = scl_threshold {
    //         unsafe { w.scl_filter_thres().bits(threshold) };
    //     }
    //     w.scl_filter_en().bit(scl_threshold.is_some())
    // });
    // } else {
    register_block.filter_cfg().modify(|_, w| {
        if let Some(threshold) = sda_threshold {
            unsafe { w.sda_filter_thres().bits(threshold) };
        }
        if let Some(threshold) = scl_threshold {
            unsafe { w.scl_filter_thres().bits(threshold) };
        }
        w.sda_filter_en().bit(sda_threshold.is_some());
        w.scl_filter_en().bit(scl_threshold.is_some())
    });
    // }
}

/// reason for receive callback being called
///
/// Needed to differentiate if the data is a partial data (watermark) or full fifo (stretch) or the final data.
pub enum SlaveEvent {
    /// called due to rx fifo watermark interrupt
    /// including the offset of how many bytes were read in the current transaction (without data from that call), kind of an offset
    RxFifoWatermark(usize),
    /// called due to clock stretch active as rx fifo is full.
    /// Incl. the offset of how many bytes were read in the current transaction (without data from that call), kind of an offset
    StretchRxFull(usize),
    /// called due to clock stretch active as address matched
    /// including the offset of how many bytes were read in the current transaction (without data from that call), kind of an offset
    StretchAddrMatch(usize),
    /// called due to clock stretch active as tx fifo empty (master requests data)
    ///
    /// expectation is that no data is read, but data to write has to be provided
    StretchTxEmpty,
    /// Transfer complete (master finished writing to us/slave)
    /// including the offset of how many bytes were read in the current transaction (without data from that call), kind of an offset
    TransComplete(usize),
}

/// type alias for the slave event callback called on read received/write requests from master
///
/// it gets the Reason, any received data from master/fifo and a buffer to write data to be sent to master.
/// Must return the number of bytes written to the buffer.
pub type I2cSlaveEventCallback =
    &'static (dyn Fn(SlaveEvent, &[u8], &mut [u8]) -> usize + Send + Sync);

/// Peripheral state for an I2C instance.
#[doc(hidden)]
#[non_exhaustive]
pub struct State {
    /// Waker for the asynchronous operations.
    pub waker: AtomicWaker,
    pub callback: RwLock<CriticalSectionRawMutex, Option<I2cSlaveEventCallback>>,
    /// how many bytes were read in the current transaction? (mainly for rxfull handling)
    pub cur_transaction_rx_cnt: AtomicUsize,
}

/// A peripheral singleton compatible with the I2C slave driver.
pub trait Instance: crate::private::Sealed + any::Degrade {
    #[doc(hidden)]
    /// Returns the peripheral data and state describing this instance.
    fn parts(&self) -> (&Info, &State);

    /// Returns the peripheral data describing this instance.
    #[doc(hidden)]
    #[inline(always)]
    fn info(&self) -> &Info {
        self.parts().0
    }

    /// Returns the peripheral state for this instance.
    #[doc(hidden)]
    #[inline(always)]
    fn state(&self) -> &State {
        self.parts().1
    }
}

for_each_i2c_slave!(
    ($inst:ident, $peri:ident, $scl:ident, $sda:ident) => {
        impl Instance for crate::peripherals::$inst<'_> {
            fn parts(&self) -> (&Info, &State) {
                #[handler]
                #[ram]
                pub(super) fn irq_handler() {
                    async_handler(&PERIPHERAL, &STATE);
                }

                static STATE: State = State {
                    waker: AtomicWaker::new(),
                    callback: RwLock::new(None),
                    cur_transaction_rx_cnt: AtomicUsize::new(0),
                };

                static PERIPHERAL: Info = Info {
                    register_block: crate::peripherals::$inst::ptr(),
                    peripheral: crate::system::Peripheral::$peri,
                    async_handler: irq_handler,
                    scl_output: OutputSignal::$scl,
                    scl_input: InputSignal::$scl,
                    sda_output: OutputSignal::$sda,
                    sda_input: InputSignal::$sda,
                };
                (&PERIPHERAL, &STATE)
            }
        }
    };
);
/// Peripheral data describing a particular I2C instance.
#[doc(hidden)]
#[derive(Debug)]
#[non_exhaustive]
pub struct Info {
    /// Pointer to the register block for this I2C instance.
    ///
    /// Use [Self::register_block] to access the register block.
    pub register_block: *const RegisterBlock,

    /// System peripheral marker.
    pub peripheral: crate::system::Peripheral,

    /// Interrupt handler for the asynchronous operations of this I2C instance.
    pub async_handler: InterruptHandler,

    /// SCL output signal.
    pub scl_output: OutputSignal,

    /// SCL input signal.
    pub scl_input: InputSignal,

    /// SDA output signal.
    pub sda_output: OutputSignal,

    /// SDA input signal.
    pub sda_input: InputSignal,
}

impl Info {
    /// Returns the register block for this I2C instance.
    pub fn regs(&self) -> &RegisterBlock {
        unsafe { &*self.register_block }
    }

    /// Listen for the given interrupts
    fn enable_listen(&self, interrupts: EnumSet<Event>, enable: bool) {
        let reg_block = self.regs();

        reg_block.int_ena().modify(|_, w| {
            for interrupt in interrupts {
                match interrupt {
                    Event::EndDetect => w.end_detect().bit(enable),
                    Event::TransComplete => w.trans_complete().bit(enable),
                    Event::RxFifoWatermark => w.rxfifo_wm().bit(enable),
                    Event::SlaveStretch => w.slave_stretch().bit(enable),
                    Event::Nack => w.nack().bit(enable),
                };
            }
            w
        });
    }

    fn interrupts(&self) -> EnumSet<Event> {
        let mut res = EnumSet::new();
        let reg_block = self.regs();

        let ints = reg_block.int_raw().read();

        if ints.slave_stretch().bit_is_set() {
            res.insert(Event::SlaveStretch);
        }
        if ints.end_detect().bit_is_set() {
            res.insert(Event::EndDetect);
        }
        if ints.nack().bit_is_set() {
            res.insert(Event::Nack);
        }
        if ints.trans_complete().bit_is_set() {
            res.insert(Event::TransComplete);
        }
        if ints.rxfifo_wm().bit_is_set() {
            res.insert(Event::RxFifoWatermark);
        }
        /*todo... full support for:
        start (1<<15) trans_start() ?
        unmatch = (1 << 18)
        */
        res
    }

    fn clear_interrupts(&self, interrupts: EnumSet<Event>) {
        let reg_block = self.regs();

        reg_block.int_clr().write(|w| {
            for interrupt in interrupts {
                match interrupt {
                    Event::EndDetect => w.end_detect().clear_bit_by_one(),
                    Event::TransComplete => w.trans_complete().clear_bit_by_one(),
                    Event::RxFifoWatermark => w.rxfifo_wm().clear_bit_by_one(),
                    Event::SlaveStretch => w.slave_stretch().clear_bit_by_one(),
                    Event::Nack => w.nack().clear_bit_by_one(),
                };
            }
            w
        });
    }
}

impl PartialEq for Info {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.register_block, other.register_block)
    }
}

unsafe impl Sync for Info {}

any_peripheral! {
    /// Any I2C peripheral.
    pub peripheral AnyI2c<'d> {
        #[cfg(i2c_slave_i2c0)]
        I2C0(crate::peripherals::I2C0<'d>),
        #[cfg(i2c_slave_i2c1)]
        I2C1(crate::peripherals::I2C1<'d>),
    }
}

impl Instance for AnyI2c<'_> {
    fn parts(&self) -> (&Info, &State) {
        any::delegate!(self, i2c => { i2c.parts() })
    }
}

impl AnyI2c<'_> {
    fn bind_peri_interrupt(&self, handler: interrupt::IsrCallback) {
        any::delegate!(self, i2c => { i2c.bind_peri_interrupt(handler) })
    }

    fn disable_peri_interrupt(&self) {
        any::delegate!(self, i2c => { i2c.disable_peri_interrupt() })
    }

    fn enable_peri_interrupt(&self, priority: crate::interrupt::Priority) {
        any::delegate!(self, i2c => { i2c.enable_peri_interrupt(priority) })
    }

    fn set_interrupt_handler(&self, handler: InterruptHandler) {
        self.disable_peri_interrupt();

        self.info().enable_listen(EnumSet::all(), false);
        self.info().clear_interrupts(EnumSet::all());

        self.bind_peri_interrupt(handler.handler());
        self.enable_peri_interrupt(handler.priority());
    }
}
