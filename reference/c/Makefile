# Host build of the reference C frontend, against the CMSIS-DSP subset the
# firmware vendors. __GNUC_PYTHON__ selects CMSIS-DSP's portable C path; a
# Cortex-M55 build instead defines ARM_MATH_HELIUM and adds
# -flax-vector-conversions (see firmware/Makefile).
DSP = ../../firmware/Drivers/CMSIS-DSP
DSP_SRC = $(addprefix $(DSP)/Source/, \
  TransformFunctions/arm_rfft_fast_f32.c TransformFunctions/arm_rfft_fast_init_f32.c \
  TransformFunctions/arm_cfft_f32.c TransformFunctions/arm_cfft_init_f32.c \
  TransformFunctions/arm_cfft_radix8_f32.c TransformFunctions/arm_bitreversal2.c \
  CommonTables/arm_common_tables.c CommonTables/arm_const_structs.c CommonTables/arm_mve_tables.c \
  ComplexMathFunctions/arm_cmplx_mag_f32.c)
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
CPPFLAGS += -D__GNUC_PYTHON__ -I. -I$(DSP)/Include -I$(DSP)/PrivateInclude

frontend_cli: frontend_cli.c birdnet_frontend.c birdnet_frontend.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ frontend_cli.c birdnet_frontend.c $(DSP_SRC) -lm

clean:
	rm -f frontend_cli

.PHONY: clean
