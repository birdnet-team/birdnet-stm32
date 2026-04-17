# Third-Party Licenses

This directory contains source files from third-party projects.
Each component retains its original license as described below.

---

## STMicroelectronics HAL/BSP Drivers — BSD-3-Clause

Applies to the following files:

- `Drivers/HAL_SD/stm32n6xx_hal_sd.c`, `stm32n6xx_hal_sd.h`, `stm32n6xx_hal_sd_ex.h`
- `Drivers/HAL_SD/stm32n6xx_ll_sdmmc.c`, `stm32n6xx_ll_sdmmc.h`
- `Drivers/stm32n6570_discovery_sd.c`, `stm32n6570_discovery_sd.h`
- `Drivers/FatFs/diskio.c`, `ff_gen_drv.c`, `ff_gen_drv.h`, `sd_diskio.c`, `sd_diskio.h`
- `Inc/stm32n6xx_hal_conf.h`, `stm32n6570_discovery_conf.h`, `stm32n6xx_it.h`, `main.h`
- `Inc/mcu_cache.h`, `npu_cache.h`, `misc_toolbox.h`
- `Src/stm32n6xx_it.c`, `sysmem.c`, `syscalls.c`
- `Src/mcu_cache.c`, `npu_cache.c`, `misc_toolbox.c`
- `Src/system_clock_config.c`
- `Config/sd_diskio_config.h`
- `startup_stm32n657xx.s`
- `network.c`, `network.h`

```
Copyright (c) 2023 STMicroelectronics.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## FatFs — FatFs License

Applies to `Drivers/FatFs/ff.c`, `Drivers/FatFs/ff.h`, `Config/ffconf.h`.

```
Copyright (C) 2022, ChaN, all right reserved.

FatFs module is an open source software. Redistribution and use of FatFs in
source and binary forms, with or without modification, are permitted provided
that the following condition is met:

1. Redistributions of source code must retain the above copyright notice,
   this condition and the following disclaimer.

This software is provided by the copyright holder and contributors "AS IS"
and any warranties related to this software are DISCLAIMED.
The copyright owner or contributors be NOT LIABLE for any damages caused
by use of this software.
```
