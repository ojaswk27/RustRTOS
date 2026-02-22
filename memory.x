/* Linker script for STM32F411 (Cortex-M4)
   FLASH: 512KB starting at 0x0800_0000
   RAM:   128KB starting at 0x2000_0000 */
MEMORY
{
    FLASH : ORIGIN = 0x08000000, LENGTH = 512K
    RAM   : ORIGIN = 0x20000000, LENGTH = 128K
}
