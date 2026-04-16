/**
  ******************************************************************************
  * @file    main.h
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef MAIN_H
#define MAIN_H

/* Includes ------------------------------------------------------------------*/
#include "stm32n6xx_hal.h"
#if NUCLEO_N6_CONFIG == 0
#include "stm32n6570_discovery.h"
#include "stm32n6570_discovery_bus.h"
#include "stm32n6570_discovery_xspi.h"
#else
#include "stm32n6xx_nucleo.h"
#include "stm32n6xx_nucleo_bus.h"
#include "stm32n6xx_nucleo_xspi.h"
#endif



/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
/* Exported macro ------------------------------------------------------------*/
/* Exported functions ------------------------------------------------------- */
void MX_USB1_OTG_HS_PCD_Init(void);
void Error_Handler(void);
void SystemClock_Config(void);
void SystemClock_Config_ResetClocks(void);
void SystemClock_Config_HSE(void);
void SystemClock_Config_HSI_no_overdrive(void);
void SystemClock_Config_HSI_overdrive(void);
void SystemClock_Config_HSI_400(void);
#endif /* MAIN_H */

