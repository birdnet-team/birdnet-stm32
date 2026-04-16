  /**
  ******************************************************************************
  * @file    misc_toolbox.c
  * @author  GPM/AIS Application Team
  * @brief   Collection of functions to perform main configurations in main.c
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

#include <string.h>     // Used for memset

#include "misc_toolbox.h"
#include "app_config.h"
#include "npu_cache.h"  // Used in NPU_config
#include "stm32n6xx_ll_usart.h" // Used for configuring UART

UART_HandleTypeDef UartHandle;


static uint32_t get_risaf_max_addr(RISAF_TypeDef *risaf)
{
  uint32_t max_addr = 0U;
  if      ((risaf == RISAF1_S)  || (risaf == RISAF1_NS))  {max_addr = RISAF1_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF2_S)  || (risaf == RISAF2_NS))  {max_addr = RISAF2_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF3_S)  || (risaf == RISAF3_NS))  {max_addr = RISAF3_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF4_S)  || (risaf == RISAF4_NS))  {max_addr = RISAF4_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF5_S)  || (risaf == RISAF5_NS))  {max_addr = RISAF5_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF6_S)  || (risaf == RISAF6_NS))  {max_addr = RISAF6_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF7_S)  || (risaf == RISAF7_NS))  {max_addr = RISAF7_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF8_S)  || (risaf == RISAF8_NS))  {max_addr = RISAF8_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF9_S)  || (risaf == RISAF9_NS))  {max_addr = RISAF9_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF11_S) || (risaf == RISAF11_NS)) {max_addr = RISAF11_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF12_S) || (risaf == RISAF12_NS)) {max_addr = RISAF12_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF13_S) || (risaf == RISAF13_NS)) {max_addr = RISAF13_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF14_S) || (risaf == RISAF14_NS)) {max_addr = RISAF14_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF15_S) || (risaf == RISAF15_NS)) {max_addr = RISAF15_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF21_S) || (risaf == RISAF21_NS)) {max_addr = RISAF21_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF22_S) || (risaf == RISAF22_NS)) {max_addr = RISAF22_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF23_S) || (risaf == RISAF23_NS)) {max_addr = RISAF23_LIMIT_ADDRESS_SPACE_SIZE;}
  return max_addr;
}

static void set_risaf_default(RISAF_TypeDef *risaf)
{
  RISAF_BaseRegionConfig_t risaf_conf;  
  risaf_conf.StartAddress = 0x0;
  risaf_conf.EndAddress   = get_risaf_max_addr(risaf); /* as the default config */
  risaf_conf.Filtering    = RISAF_FILTER_ENABLE; // Base region enable (otherwise access control is secure, privileged, trusted domain CID = 1)
  risaf_conf.PrivWhitelist  = RIF_CID_NONE; // apps running in all compartments can access to region in priv/unpriv mode
  risaf_conf.ReadWhitelist  = RIF_CID_MASK; // apps running in all compartments can R in this region
  risaf_conf.WriteWhitelist = RIF_CID_MASK; // apps running in all compartments can W in this region
  // Configure 2 regions with this config, fully overlapping, one for secure one for non secure accesses:
  risaf_conf.Secure = RIF_ATTRIBUTE_SEC;    // Only secure requests can access this region
  HAL_RIF_RISAF_ConfigBaseRegion(risaf, 0, &risaf_conf);
  risaf_conf.Secure = RIF_ATTRIBUTE_NSEC;    // Only non-secure requests can access this region
  HAL_RIF_RISAF_ConfigBaseRegion(risaf, 1, &risaf_conf);
}

#ifdef HAL_BSEC_MODULE_ENABLED
static void fuse_hardware_conf(uint32_t bit_to_fuse)
{
  uint32_t fuse_id, data, fuse_mask;
  BSEC_HandleTypeDef sBsecHandler;
  sBsecHandler.Instance = BSEC;
  fuse_mask = (1U << bit_to_fuse);
  /* Read current value of fuse */
  fuse_id = 124U; // HCONF1 OTP (see reference manual)
  if (HAL_BSEC_OTP_Read(&sBsecHandler, fuse_id, &data) == HAL_OK)
  {
    /* Check if bit has already been set */
    if ((data & fuse_mask) != fuse_mask)
    {
      data |= fuse_mask;
      /* Bitwise programming of lower bits */
      if (HAL_BSEC_OTP_Program(&sBsecHandler, fuse_id, data, HAL_BSEC_NORMAL_PROG) == HAL_OK)
      {
        /* Read lower bits to verify the correct programming */
        if (HAL_BSEC_OTP_Read(&sBsecHandler, fuse_id, &data) == HAL_OK)
        {
          if ((data & fuse_mask) != fuse_mask)
          {
            /* Error : Fuse programming not taken in account */
            while(1){};
          }
        }
        else
        {
          /* Error : Fuse read unsuccessful */
          while(1){};
        }
      }
      else
      {
        /* Error : Fuse programming unsuccessful */
        while(1){};
      }
    }
  }
  else
  {
    /* Error  : Fuse read unsuccessful */
    while(1){};
  }
}
#endif

void set_clk_sleep_mode(void)
{
  /* Leave clocks enabled in Low Power modes */
  // Low-power clock enable misc
  __HAL_RCC_DBG_CLK_SLEEP_ENABLE();
  __HAL_RCC_XSPIPHYCOMP_CLK_SLEEP_ENABLE();
  
  // Low-power clock enable for memories
  __HAL_RCC_AXISRAM1_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM2_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM3_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM4_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM5_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM6_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_FLEXRAM_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_CACHEAXIRAM_MEM_CLK_SLEEP_ENABLE();
  // LP clock AHB1: None
  // LP clock AHB2: None
  // LP clock AHB3
  __HAL_RCC_RIFSC_CLK_SLEEP_ENABLE();
  __HAL_RCC_RISAF_CLK_SLEEP_ENABLE();
  __HAL_RCC_IAC_CLK_SLEEP_ENABLE();
  // LP clock AHB4: None
  // LP clocks AHB5
  __HAL_RCC_XSPI1_CLK_SLEEP_ENABLE();
  __HAL_RCC_XSPI2_CLK_SLEEP_ENABLE();
  __HAL_RCC_CACHEAXI_CLK_SLEEP_ENABLE();
  __HAL_RCC_NPU_CLK_SLEEP_ENABLE();
  // LP clocks APB1: None
  // LP clocks APB2
  __HAL_RCC_USART1_CLK_SLEEP_ENABLE();
  // LP clocks APB4: None
  // LP clocks APB5: None
}

/* Change the VDDCORE level for overdrive modes
 * (Nucleo, legacy DK -before rev. C)
 * Using the I2c to control the Step-Down Converter 
 *      This is mandatory / safer if an "overdrive" configuration is needed
 *      For the DK <rev.C /Nucleo, step-down converter = TPS62864
 *      Setting resistor = 56.2 kohm
 *              with 56.2kOhm: Output level: 0.80 V 
 *              with 56.2kOhm: I2C device Address : 1001 001 = 0x49
 * (DK -after rev. C included)
 * Using the GPIO to control the step-down converter (for DK rev >= C)
 */
void upscale_vddcore_level(void)
{
#if ((NUCLEO_N6_CONFIG == 0) && defined(STM32N6570_DK_REV) && (STM32N6570_DK_REV>=STM32N6570_DK_C01))      // Handle new DK boards with new SMPS controlled by GPIO  
  BSP_SMPS_Init(SMPS_VOLTAGE_OVERDRIVE);
#else   // Handle Nucleo boards or DK boards before rev.C
  uint8_t tmp;
  tmp = 0x64;  // 0x64 is 900mV, LSB=5mV
  BSP_I2C2_Init();
  // Address of the device on 7 bits: API requires the address to be switched left by 1
  // Write tmp on register 0x1 (Vout register 1), length=1
  BSP_I2C2_WriteReg(0x49 << 1, 0x01, &tmp, 1);  
#endif
  HAL_Delay(1); /* Assuming Voltage Ramp Speed of 1mV/us --> 100mV increase takes 100us */
}

/* Initialises UART @  USE_UART_BAUDRATE
 * Configures GPIO pins for UART (
 * Enables clocks for UART/GPIO
 */
void UART_Config(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  
  /* Peripheral clock enable */
  __HAL_RCC_USART1_CLK_ENABLE();
  __HAL_RCC_USART1_FORCE_RESET();
  __HAL_RCC_USART1_RELEASE_RESET();
  __HAL_RCC_GPIOE_CLK_ENABLE();

    /**USART1 GPIO Configuration
   * PE5     ------> USART1_TX
   * PE6     ------> USART1_RX
   */
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_6;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
  
  /* Peripheral config */
  UartHandle.Instance = USART1;
  UartHandle.Init.BaudRate = USE_UART_BAUDRATE;
  UartHandle.Init.WordLength = UART_WORDLENGTH_8B;
  UartHandle.Init.StopBits = UART_STOPBITS_1;
  UartHandle.Init.Parity = UART_PARITY_NONE;
  UartHandle.Init.Mode = UART_MODE_TX_RX;
  UartHandle.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  UartHandle.Init.OverSampling = UART_OVERSAMPLING_16;
  UartHandle.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  UartHandle.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  UartHandle.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&UartHandle) != HAL_OK)
  {
    while(1){};
  }
  // Disable FIFO mode
  UartHandle.Instance->CR1 &= ~USART_CR1_FIFOEN;
}

/**
* @brief  USB RIF Configuration.
* @retval None
*/
void USB_RIF_Config(void)
{
  RIMC_MasterConfig_t     RIMC_master = {0};

  __HAL_RCC_RIFSC_CLK_ENABLE();

  RIMC_master.MasterCID = RIF_CID_1;
  RIMC_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;

  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_OTG1, &RIMC_master);
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_OTG2, &RIMC_master);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_OTG1HS, RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_OTG2HS, RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
}

void NPU_Config(void)
{
  // Enable NPU
  __HAL_RCC_NPU_CLK_ENABLE();
  __HAL_RCC_NPU_FORCE_RESET();
  __HAL_RCC_NPU_RELEASE_RESET();
  // Enable Cache-AXI
  __HAL_RCC_CACHEAXI_CLK_ENABLE();
  __HAL_RCC_CACHEAXI_FORCE_RESET();
  __HAL_RCC_CACHEAXI_RELEASE_RESET();
  
  // __HAL_RCC_CACHEAXI_CLK_SLEEP_DISABLE();
  // __HAL_RCC_NPU_CLK_SLEEP_DISABLE();
  // __HAL_RCC_RAMCFG_CLK_SLEEP_DISABLE();
  npu_cache_init();
  
#ifdef USE_NPU_CACHE
   npu_cache_enable(); // Useless: already enabled by init
#else
   npu_cache_disable();
#endif

  RIMC_MasterConfig_t master_conf;
  /* Enable Secure access for NPU */
  master_conf.MasterCID = RIF_CID_1;    // Master CID = 1
  master_conf.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV; // Priviledged secure
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &master_conf);  
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU, RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);
  
}

void RISAF_Config(void)
{
  /*
  *  Note: before to set a risaf for a given IP, the IP
  *        should be clocked.
  */
  set_risaf_default(RISAF2_S);          /* SRAM1_AXI */
  set_risaf_default(RISAF3_S);          /* SRAM2_AXI */
  
  set_risaf_default(RISAF4_S);          /* NPU MST0 */
  set_risaf_default(RISAF5_S);          /* NPU MST1 */
  
  set_risaf_default(RISAF6_S);          /* SRAM3,4,5,6_AXI */
  set_risaf_default(RISAF7_S);          /* FLEXMEM */
  
#ifdef USE_NPU_CACHE
  set_risaf_default(RISAF8_S);          /* NPU_CACHE */
  set_risaf_default(RISAF15_S);         /* NPU_CACHE config */
#endif  
  
  // set_risaf_default(RISAF9_S);       /* VENC */
  
#if defined(USE_EXTERNAL_MEMORY_DEVICES) && USE_EXTERNAL_MEMORY_DEVICES == 1
#if (NUCLEO_N6_CONFIG == 0)
  set_risaf_default(RISAF11_S);         /* OCTOSPI1 0x9000 0000 */
#endif
  set_risaf_default(RISAF12_S);         /* OCTOSPI2 0x7000 0000 */
  // set_risaf_default(RISAF13_S);      /* OCTOSPI3 0x8000 0000 */
#endif
  
}


void set_vector_table_addr(void)
{
  __disable_irq();
  SCB->VTOR = 0x34000000;   // USED WITH no FLEXMEM Extension (standard scenario)
  //SCB->VTOR = 0x10000000; // USED WITH FLEXMEM Extension  (execution from ITCM only)
  /* Set default Vector Table location after system reset or return from Standby */
  //SYSCFG->INITSVTORCR = SCB->VTOR;
  __DSB();
  memset((uint32_t *)NVIC->ICER, 0xFF, sizeof(NVIC->ICER)); // Disable all irq (IRQ 139 is enabled by default)
  memset((uint32_t *)NVIC->ICPR, 0xFF, sizeof(NVIC->ICPR)); // Clear pending IRQs (LPTIM4 has a pending IRQ when exiting bootrom)
  __enable_irq();
}


void system_init_post(void)
{  
  __HAL_RCC_SYSCFG_CLK_ENABLE();
  __HAL_RCC_CRC_CLK_ENABLE();
   
  /* Enable NPU RAMs (4x448KB) + CACHEAXI */
  RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN | RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN;
  RCC->MEMENR |= RCC_MEMENR_CACHEAXIRAMEN; // RCC_MEMENR_NPUCACHERAMEN;
  
  RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
  RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
  RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
  RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
  RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    
  /* Allow caches to be activated. Default value is 1, but the current boot sets it to 0 */
  MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_DCACTIVE_Msk | MEMSYSCTL_MSCR_ICACTIVE_Msk;
}

#ifdef HAL_BSEC_MODULE_ENABLED
void fuse_vddio(void)
{
    // Fuse bit for VDDIO2 (HSLV_VDDIO2): used for PSRAM / XSPIM 1
    fuse_hardware_conf(16);
    // Fuse bit for VDDIO3 (HSLV_VDDIO3): used for external Flash / XSPIM2
    fuse_hardware_conf(15);
}
#endif