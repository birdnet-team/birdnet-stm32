/**
  ******************************************************************************
  * @file    Inc/misc_toolbox.h
  * @author  MCD Application Team
  * @brief   Header for main.c module
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
#ifndef MISC_TOOLBOX_H
#define MISC_TOOLBOX_H
/* Includes ------------------------------------------------------------------*/
#include "stm32n6xx_hal.h"
#if NUCLEO_N6_CONFIG == 0
#include "stm32n6570_discovery.h"
#include "stm32n6570_discovery_bus.h"
#include "stm32n6570_discovery_xspi.h"
#else
#include "stm32n6xx_nucleo.h"
#include "stm32n6xx_nucleo_bus.h"    // No implementation of the I2c for nucleo, the dk implem is coherent
#include "stm32n6xx_nucleo_xspi.h"
#endif

/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
extern UART_HandleTypeDef UartHandle;
/* Exported macro ------------------------------------------------------------*/
/* Exported functions ------------------------------------------------------- */

// Sets clock mode in sleep mode (everything is ON)
void set_clk_sleep_mode(void);

// Make VDDCore higher (to be used when overdriving the MCU)
void upscale_vddcore_level(void);

// Configures UART
void UART_Config(void);

// Configures the NPU: Activates clocks, activate cache and setup AXI bus Master/Slave
void NPU_Config(void);

// Configurares the USB RIF
void USB_RIF_Config(void);

// Configures the RISAFs: Everything that is used is set to "passthrough"
void RISAF_Config(void);

// Quick function to set the vector table address
void set_vector_table_addr(void);

// Basic initialization of the board after startup: Activates all RAMs, allow caches to be enabled.
void system_init_post(void);

#ifdef HAL_BSEC_MODULE_ENABLED
// Fuse OTP bits to set VDDIO2/3 I/O segments below 2.5V for I/O mode
void fuse_vddio(void);
#endif

#endif // MISC_TOOLBOX_H