import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    power = 1 # percent of laser power

    # Pulses are square waves
    # They have a period, they can project 
    # pulses (= number of flashes)
    period = 1000 # [ms]
    pulse_duration = .5 # [s]
    interpulse_wait = .5 # [s]
    pulses = 4
    repetitions = 10
    start = 0
    initial_delay = 10 # in units of period = number of periods to wait

    roi_on = 'on.png' # image to be loaded

    # pulse paradigm (only for visualization - double checking)
    # sampling = 10**4
    # time = np.arange(0, period, 1) # 1ms step samplign time
    # flash = np.zeros((len(time)))
    # for i in range(pulses):
    #     flash[
    #         i * (pulse_duration + interpulse_wait):
    #         i * (pulse_duration + interpulse_wait) + pulse_duration
    #         ] = 1
    # with plt.xkcd():
    #     plt.figure('oneperiod')
    #     plt.plot(time, flash)
    #     plt.show()

    # any pulse counts as a line. Any wait counts as a line.
    lines_to_write = repetitions * 2 * pulses
    # first lines in the scripts must be these:
    head = '\n'
    head += '[DMD]\n'
    head += 'MGI RWA Section Options=2.0.1'
    head += ' %04Y%02m%02d %02H%02M%S%25u*~^[|.^W%d*~^[|.^W,*~^[|.^W%#_13g\n'
    head += '[DMD script steps]=<' + str(lines_to_write) + '>\n'

    waves = [50, 100, 150]
    with open('/home/ngc/Desktop/test.dmdsc', 'w') as script:
        script.write(head)
        for i in waves:
            incipit = (
                '[DMD script steps] '
                + str(i)
                + '.DMD settings for 1 TM.'
                ) 
            signal = incipit + 'PNG name=' + roi_on + '\n'
            signal += incipit + 'TM index=' + str(i) + '\n'
            signal += incipit + 'Pre wait (s)=0\n'
            signal += incipit + 'Pulse (s)=' + str(pulse_duration) + '\n'
            signal += incipit + '# triggers=' + str(pulses) + '\n'
            signal += incipit + 'Period (s)=' \
                + str(pulse_duration + interpulse_wait) + '\n'
            signal += incipit + 'Final Offset (pix).X=0\n'
            signal += incipit + 'Final Offset (pix).Y=0\n'
            signal += incipit + 'Motion?=FALSE\n'
            signal += incipit + 'Offset space=DMD\n'
            signal += incipit + 'Power Multiplier=' + str(power) + '\n'
            signal += '\n'
            script.write(signal)

                






