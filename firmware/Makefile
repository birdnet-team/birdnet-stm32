GCC_PATH ?=
TARGET = Project
BUILD_CONF ?= N6-DK
OPT = -Os -g3

# --- Paths
XCUBE_AI_DIR ?= /home/mi/Code/X-CUBE-AI.10.2.0
MIDDLEWARES_PATH = $(XCUBE_AI_DIR)/Middlewares/ST
ATONN_RT_PATH = $(MIDDLEWARES_PATH)/AI/Npu/ll_aton
BSP_PATH = $(XCUBE_AI_DIR)/Projects/STM32N6570-DK/Applications/NPU_Validation/Drivers/BSP
CMSIS_PATH = $(XCUBE_AI_DIR)/Projects/STM32N6570-DK/Applications/NPU_Validation/Drivers/CMSIS
N6_DRIVER_PATH = $(XCUBE_AI_DIR)/Projects/STM32N6570-DK/Applications/SNS/Drivers/STM32N6xx_HAL_Driver
DK_DRIVER_PATH = $(BSP_PATH)/STM32N6570-DK

# --- Sources
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton_cipher.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton_dbgtrc.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton_debug.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton_lib.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton_lib_sw_operators.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton_rt_main.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton_runtime.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_aton_util.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_sw_float.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ll_sw_integer.c
ATONN_SOURCES += $(ATONN_RT_PATH)/ecloader.c

DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_bsec.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_cacheaxi.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_cortex.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_gpio.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_i2c.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_i2c_ex.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_pwr.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_pwr_ex.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_rcc.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_rcc_ex.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_rif.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_uart.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_uart_ex.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_xspi.c

C_SOURCES += Src/main.c
C_SOURCES += Src/stm32n6xx_it.c
C_SOURCES += Src/system_clock_config.c
C_SOURCES += Src/misc_toolbox.c
C_SOURCES += Src/npu_cache.c
C_SOURCES += Src/mcu_cache.c
C_SOURCES += Src/sysmem.c
C_SOURCES += Src/syscalls.c
C_SOURCES += network.c
C_SOURCES += $(ATONN_SOURCES)
C_SOURCES += $(DRIVER_SOURCES)
C_SOURCES += $(BSP_PATH)/STM32N6570-DK/stm32n6570_discovery.c
C_SOURCES += Drivers/HAL_SD/stm32n6xx_hal_sd.c
C_SOURCES += Drivers/HAL_SD/stm32n6xx_ll_sdmmc.c
C_SOURCES += Drivers/stm32n6570_discovery_sd.c
FATFS_PATH = FatFs
C_SOURCES += Src/wav_reader.c
C_SOURCES += Src/audio_stft.c
C_SOURCES += Src/sd_handler.c
C_SOURCES += Src/fft.c
C_SOURCES += Drivers/$(FATFS_PATH)/ff.c
C_SOURCES += Drivers/$(FATFS_PATH)/diskio.c
C_SOURCES += Drivers/$(FATFS_PATH)/ff_gen_drv.c
C_SOURCES += Drivers/$(FATFS_PATH)/sd_diskio.c

ASM_SOURCES = startup_stm32n657xx.s
C_SOURCES += $(CMSIS_PATH)/Device/ST/STM32N6xx/Source/Templates/system_stm32n6xx_fsbl.c

PREFIX = arm-none-eabi-

ifdef GCC_PATH
CC = $(GCC_PATH)/$(PREFIX)gcc
AS = $(GCC_PATH)/$(PREFIX)gcc -x assembler-with-cpp
CP = $(GCC_PATH)/$(PREFIX)objcopy
SZ = $(GCC_PATH)/$(PREFIX)size
else
CC = $(PREFIX)gcc
AS = $(PREFIX)gcc -x assembler-with-cpp
CP = $(PREFIX)objcopy
SZ = $(PREFIX)size
endif
HEX = $(CP) -O ihex
BIN = $(CP) -O binary -S

CPU = -mcpu=cortex-m55
FPU = -mfpu=fpv5-sp-d16
FLOAT-ABI = -mfloat-abi=hard
MCU = $(CPU) -mthumb $(FPU) $(FLOAT-ABI) -mcmse

C_DEFS = -DSTM32N657xx -DUSE_HAL_DRIVER -DCORE_CM55 -DSECURE=1 -DCPU_IN_SECURE_STATE -DUSE_UART_BAUDRATE=921600 -DUSE_OVERDRIVE=0 -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_BARE_METAL -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_EB_DBG_INFO -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -DLL_ATON_SW_FALLBACK=1 

C_INCLUDES += -IInc -I. -IDrivers/FatFs -IConfig -IDrivers -IDrivers/HAL_SD
C_INCLUDES += -I$(CMSIS_PATH)/Core/Include
C_INCLUDES += -I$(CMSIS_PATH)/Device/ST/STM32N6xx/Include
C_INCLUDES += -I$(CMSIS_PATH)/Device/ST/STM32N6xx/Include/Templates
C_INCLUDES += -I$(N6_DRIVER_PATH)/Inc
C_INCLUDES += -I$(N6_DRIVER_PATH)/Inc/Legacy
C_INCLUDES += -I$(DK_DRIVER_PATH)
C_INCLUDES += -I$(ATONN_RT_PATH)
C_INCLUDES += -I$(MIDDLEWARES_PATH)/AI/Inc
C_INCLUDES += -I$(MIDDLEWARES_PATH)/AI/Npu/Devices/STM32N6XX

CFLAGS = $(MCU) $(C_DEFS) $(C_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections

ASFLAGS = $(MCU) $(OPT) -Wall -fdata-sections -ffunction-sections

LDSCRIPT = STM32N657xx.ld
LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections /home/mi/Code/X-CUBE-AI.10.2.0/Middlewares/ST/AI/Lib/GCC/ARMCortexM55/NetworkRuntime1020_CM55_GCC.a

BUILD_DIR = build

all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).hex $(BUILD_DIR)/$(TARGET).bin

OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
vpath %.c $(sort $(dir $(C_SOURCES)))
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.s=.o)))
vpath %.s $(sort $(dir $(ASM_SOURCES)))

$(BUILD_DIR)/%.o: %.c Makefile | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) -c $(CFLAGS) -Wa,-a,-ad,-alms=$(BUILD_DIR)/$(notdir $(<:.c=.lst)) $< -o $@

$(BUILD_DIR)/%.o: %.s Makefile | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(AS) -c $(CFLAGS) $< -o $@

$(BUILD_DIR)/$(TARGET).elf: $(OBJECTS) Makefile
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	$(SZ) $@

$(BUILD_DIR)/%.hex: $(BUILD_DIR)/%.elf | $(BUILD_DIR)
	$(HEX) $< $@

$(BUILD_DIR)/%.bin: $(BUILD_DIR)/%.elf | $(BUILD_DIR)
	$(BIN) $< $@

$(BUILD_DIR):
	mkdir $@

clean:
	-rm -fR $(BUILD_DIR)
