//*****************************************************************************
// Tiva 
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "inc/hw_memmap.h"
#include "inc/hw_ints.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/uart.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/pwm.h"
#include "driverlib/timer.h"


//====================================================
#define GPIOB_BASE  GPIO_PORTB_BASE
#define PERIODO_PWM 120000 

volatile uint8_t estado = 0;

volatile uint32_t ciclos_timer = 0; 

//===========================================
void Timer1AIntHandler(void)
{
    TimerIntClear(TIMER1_BASE, TIMER_TIMA_TIMEOUT);
    GPIOPinWrite(GPIOB_BASE, 0x30, 0x00);
    estado = 3; 
    ciclos_timer = 3 * 120000000; // Si no hay nada, la próxima será de 3s
}

//=========================================
void UART3IntHandler(void)
{
    uint32_t status = UARTIntStatus(UART3_BASE, true);
    UARTIntClear(UART3_BASE, status);

    while(UARTCharsAvail(UART3_BASE)) {
        char c = UARTCharGet(UART3_BASE);

        if(c == 'c'){
            estado = 1;
            GPIOPinWrite(GPIOB_BASE, 0x30, 0x10); 
            TimerDisable(TIMER1_BASE, TIMER_A);
            TimerLoadSet(TIMER1_BASE, TIMER_A, ciclos_timer - 1); 
            TimerEnable(TIMER1_BASE, TIMER_A);
            ciclos_timer = 120000000; 
        }
        else if(c == 's'){ // SALVIETTI
            estado = 2;
            GPIOPinWrite(GPIOB_BASE, 0x30, 0x20); 
            TimerDisable(TIMER1_BASE, TIMER_A);
            TimerLoadSet(TIMER1_BASE, TIMER_A, ciclos_timer - 1); 
            TimerEnable(TIMER1_BASE, TIMER_A);
            ciclos_timer = 120000000;
        }
        else if(c == 'p' || c == 'f'){ 
            estado = 5; 
        }
        else if(c == 'm'){ 
            estado = 4;
            TimerDisable(TIMER1_BASE, TIMER_A);   
            GPIOPinWrite(GPIOB_BASE, 0x30, 0x00); 
        }
    }
}


int main(void)
{
    ciclos_timer = 120000000;
    uint32_t reloj = SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ | SYSCTL_OSC_MAIN | SYSCTL_USE_PLL | SYSCTL_CFG_VCO_480), 120000000);

    //  Periféricos
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_UART3);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_PWM0);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER1);

    // UART
    GPIOPinConfigure(GPIO_PA4_U3RX);
    GPIOPinConfigure(GPIO_PA5_U3TX);
    GPIOPinTypeUART(GPIO_PORTA_BASE, 0x30);
    UARTConfigSetExpClk(UART3_BASE, reloj, 9600, (UART_CONFIG_WLEN_8 | UART_CONFIG_STOP_ONE | UART_CONFIG_PAR_NONE));
    
    // PWM
    GPIOPinConfigure(GPIO_PF1_M0PWM1);
    GPIOPinTypePWM(GPIO_PORTF_BASE, GPIO_PIN_1);
    PWMGenConfigure(PWM0_BASE, PWM_GEN_0, PWM_GEN_MODE_DOWN);
    PWMGenPeriodSet(PWM0_BASE, PWM_GEN_0, PERIODO_PWM);
    PWMPulseWidthSet(PWM0_BASE, PWM_OUT_1, PERIODO_PWM / 2); 
    PWMOutputState(PWM0_BASE, PWM_OUT_1_BIT, false);
    PWMGenEnable(PWM0_BASE, PWM_GEN_0);

    // GPIO
    GPIOPinTypeGPIOOutput(GPIOB_BASE, 0x30);

    // Timer
    TimerConfigure(TIMER1_BASE, TIMER_CFG_ONE_SHOT); 
    TimerIntEnable(TIMER1_BASE, TIMER_TIMA_TIMEOUT);
    IntEnable(INT_TIMER1A);

    // Interrupciones
    UARTIntEnable(UART3_BASE, UART_INT_RX);
    IntEnable(INT_UART3);
    IntMasterEnable();

    while(1)
    {
        if (estado == 4) {
            PWMOutputState(PWM0_BASE, PWM_OUT_1_BIT, true);
        } else {
            PWMOutputState(PWM0_BASE, PWM_OUT_1_BIT, false);
        }
    }
}