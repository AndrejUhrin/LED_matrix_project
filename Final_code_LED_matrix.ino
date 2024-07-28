#include <LED_Matrix_2_second_recordings_final_inferencing.h>
#include <Arduino.h>
#include <PDM.h>
#include <Wire.h>
#include "grove_two_rgb_led_matrix.h"

#ifdef ARDUINO_SAMD_VARIANT_COMPLIANCE
#define SERIAL SerialUSB
#else
#define SERIAL Serial
#endif

#define DISPLAY_COLOR_RED  0xFF0000  

GroveTwoRGBLedMatrixClass matrix;


typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[4096]; 
static bool debug_nn = false; 

void waitForMatrixReady() {
    delay(1000);
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead >> 1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

    if (inference.buffer == NULL) {
        return false;
    }

    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize(8192);  

    
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        microphone_inference_end();
        return false;
    }

    // 
    PDM.setGain(127);

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    inference.buf_ready = 0;
    inference.buf_count = 0;

    while (inference.buf_ready == 0) {
        delay(10);
    }

    return true;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffer);
}

void setup()
{
    Wire.begin();
    SERIAL.begin(115200);
    while (!SERIAL);  
    SERIAL.println("Setup started");

    waitForMatrixReady();
    uint16_t VID = matrix.getDeviceVID();
    SERIAL.print("VID: 0x");
    SERIAL.println(VID, HEX);
    if (VID != 0x2886) {
        SERIAL.println("Cannot detect LED matrix!!!");
        while (1);
    }
    SERIAL.println("Matrix init success!!!");

    matrix.displayColorBlock(0x000000, 0, true);  
    SERIAL.println("Matrix cleared");

    
    if (!microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT)) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }

    SERIAL.println("PDM microphone initialized");
}

uint64_t ghost[] = {
    0xffff00000000ffff,
    0xff000000000000ff,
    0xff00fefe0000fefe,
    0x0000feac0000feac,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0xff00ff00ff00ff00
};

String last_command = "novoice";
unsigned long emoji_start_time = 0;
int current_emoji_index = 0;
bool emoji_mode = false;
bool display_active = false;

void displayCommand(String command) {
    if (command == "matrix") {
        emoji_mode = false;
        matrix.displayColorBlock(0x000000, 0, true);  
        display_active = false;
    } else if (!display_active) {
        if (command == "red") {
            matrix.displayColorBlock(DISPLAY_COLOR_RED, 0, true);  
            display_active = true;
        } else if (command == "emoji") {
            emoji_mode = true;
            emoji_start_time = millis();
            current_emoji_index = 0;
            display_active = true;
        } else if (command == "ghost") {
            matrix.displayFrames(ghost, 0, true, 1);  
            display_active = true;
        }
    }
}

void loop()
{
    
    if (emoji_mode) {
        unsigned long current_time = millis();
        if (current_time - emoji_start_time >= 1000) {  
            emoji_start_time = current_time;
            current_emoji_index++;
            if (current_emoji_index >= 35) {
                current_emoji_index = 0;
            }
            matrix.displayEmoji(current_emoji_index, 0, true);
        }
    }

    
    ei_printf("Starting inferencing...\n");

    ei_printf("Recording...\n");
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }
    ei_printf("Recording done\n");

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": \n");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);

        if (result.classification[ix].value > 0.8) {
            String current_command = result.classification[ix].label;
            if (current_command != last_command && current_command != "novoice") {
                SERIAL.println("Command recognized: " + current_command);
                displayCommand(current_command);
                last_command = current_command;
            }
        }
    }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

    delay(2000);  
}