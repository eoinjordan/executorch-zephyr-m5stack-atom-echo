/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum led_state {
	LED_OFF,
	LED_LISTENING, /* dim blue */
	LED_WAKE,      /* green */
	LED_MUTED,     /* dim purple */
	LED_ERROR,     /* red */
};

int led_init(void);
void led_set(enum led_state s);

#ifdef __cplusplus
}
#endif
