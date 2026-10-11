/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * The ATOM's single SK6812 on GPIO27, through Zephyr's ws2812-spi led_strip
 * driver that the m5stack_atom_lite board already defines.
 */

#include "led.h"

#include <zephyr/device.h>
#include <zephyr/drivers/led_strip.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

static const struct device *const strip = DEVICE_DT_GET(DT_ALIAS(led_strip));

int led_init(void)
{
	if (!device_is_ready(strip)) {
		printk("led: strip not ready\n");
		return -ENODEV;
	}
	led_set(LED_OFF);
	return 0;
}

void led_set(enum led_state s)
{
	struct led_rgb px = {0};

	switch (s) {
	case LED_LISTENING:
		px.b = 24;
		break;
	case LED_WAKE:
		px.g = 160;
		break;
	case LED_MUTED:
		px.r = 20;
		px.b = 20;
		break;
	case LED_ERROR:
		px.r = 120;
		break;
	default:
		break;
	}
	if (device_is_ready(strip)) {
		led_strip_update_rgb(strip, &px, 1);
	}
}
