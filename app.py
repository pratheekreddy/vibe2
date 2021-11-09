from flask import Flask, flash, render_template, redirect, url_for, request, jsonify
import json
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import statistics
from scipy.signal import find_peaks, spectrogram
from flask_cors import CORS
import wave

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

print("Initializing Server...")

# the binary code to send to the app
# Switch for when logon is triggered
loginAttempt = False

collectionTimeStamp = ""

# code = None
fakeCode = None
buff = None

authenticated = False
log_results = True

hex_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
leeway = 0.102  # Allows for values that are slightly above 0 to still be counted as 0
minimum_length = 3
trueCode = None
lookup_table = {}   # Go from duration to character
reverse_lookup_table = {}   # Go from character to duration
buffers = []
records = []


"""smooth_sample
Smooths sample. Idk how, legacy code
"""


def smooth_sample(sample, smoothing=10):
    frame = pd.DataFrame(sample).rolling(smoothing).mean()
    frame = frame.shift(periods=-smoothing // 2)
    norm = np.subtract(sample, frame.values.reshape(len(sample), ))
    return norm
"""
START OF AUDIO CODE
THIS CODE RETURNS THE ACCURACY OF THE VIBRATION
"""
def closest(x):
    arr = np.array([45,65,85,105,125,145,165,185,205,225,245,265,285,305,325,345])
    difference_array = np.absolute(arr-x)
    index = difference_array.argmin()
    return index

def accuracy(true_code,two_fa):
    distance=0
    if(len(true_code)!= len(two_fa)):
        return 0
    for i in range(0, len(true_code)):
        diff = abs(int(two_fa[i], 16) - int(true_code[i], 16))
        distance+=diff
    totalHexBits = len(true_code) * 4 
    percentCorrect = round(((totalHexBits - distance) / totalHexBits) * 100, 4)
    return percentCorrect

def visualize(file):
    
    raw = wave.open(file)

    mapping={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F'}
    
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
    f_rate = raw.getframerate()
    frames = raw.getnframes()
    duration = frames / float(f_rate)
    print(duration)
    two_fa=''

    val=len(signal)//4000
    n=len(signal)-val*4000

    for x in range(1, n+1, 1):
        signal=np.delete(signal,len(signal)-1,0)

    final=np.mean(signal.reshape(-1, val), axis=1)


    time = np.linspace(
        0, # start
        len(final),
        num = len(final)
    )

    positive=final>0
    po=final[positive]
    pavg=np.average(po)*1.7
    navg=-0.8*pavg

    #------------------
    code=[]

    for x in range(0,len(final),1):
        if(final[x]>pavg):
            code.append(x)

    first=0
    error=0
    for x in range(len(code)-1):
        if(code[x+1]-code[x]>10):
            error=error+1
            if(code[x]-code[first]>25):
                d=code[x]-code[first]
                a=closest(d)
                two_fa=two_fa+mapping[a]
            last=code[x+1]
            first=x+1

    a=closest(code[len(code)-1]-last)
    two_fa=two_fa+mapping[a]
    print(two_fa)
    return (accuracy(trueCode,two_fa))

"""
    END OF AUDIO CODE
"""
"""rand_key
Generate a random auth token.
- char_options: List containing all possible characters in the string
- code_length: Length of the token to generate
"""


def rand_key(char_options, code_length):
    key = ''.join(random.choice(char_options) for i in range(code_length))
    return key


"""round_to_nearest_ten
Round a number to a value divisible by 10. By default, is biased to round down heavily
- original_num: number to start with
- bias: positive value gives bias to round up, negative is biased to round down
"""


def round_to_nearest_ten(original_num, bias=-2.75):
    num = original_num + bias

    smaller_multiple = (num // 10) * 10
    larger_multiple = smaller_multiple + 10

    if num - smaller_multiple > larger_multiple - num:
        return int(larger_multiple)
    else:
        return int(smaller_multiple)


"""hex_table_setter
Used to set the global lookup table to be a basic hex table. Translates durations into characters and vice versa
"""


def hex_table_setter(start=40, delta=10):
    global lookup_table             # Mapping from vibration duration to hex value (e.g. '40' and '50' map to hex value '0')
    global reverse_lookup_table     # Mapping from hex value to vibration duration (e.g. hex value '0' maps to '45')
    print("Generating hex table...")
    current = start
    for hex_val in hex_set:
        for _ in range(2):
            lookup_table[current] = hex_val
            current = current + delta

    current = start + delta/2
    for hex_val in hex_set:
        reverse_lookup_table[hex_val] = current
        current = current + delta * 2


"""table_lookup
Returns the corresponding token value for a given vibration distance
"""


def table_lookup(val):
    try:
        return lookup_table[val]
    except:
        min_key = min(lookup_table.keys())
        max_key = max(lookup_table.keys())
        if val < min_key:
            return lookup_table[min_key]
        elif val > max_key:
            return lookup_table[max_key]
        else:
            return 'X'


"""buffer_counter
Works with vib_counter, counts periods of idleness in a dataset
"""


def buffer_counter(arr, curr_index, min_length):
    global buffers
    arr_length = len(arr) - 10
    counter = 0
    while curr_index < arr_length:
        val = arr[curr_index]
        if val <= 0 + leeway or val == "nan":
            counter = counter + 1
        else:
            break

        curr_index += 1

    if curr_index < arr_length:
        buffers.append(counter)
        vib_counter(arr, curr_index, min_length)


"""vib_counter
Works with buffer_counter, counts periods of vibration in a dataset
"""


def vib_counter(arr, curr_index, min_length):
    global records
    arr_length = len(arr)
    streak = 0
    counter = 0
    negatives = 0
    while streak < min_length and curr_index < arr_length:
        val = arr[curr_index]
        if (val > 0 - leeway) and val < 0 + leeway:
            streak += 1
        else:
            if val > 0:
                counter = counter + 1 + streak + negatives
                streak = 0
                negatives = 0
            else:
                negatives += 1

        curr_index += 1

    if counter > 0:
        records.append(counter)
    if curr_index < arr_length:
        buffer_counter(arr, curr_index - streak - negatives, min_length)


def process_signal(sample):
    start = time.time()
    if len(sample) == 0:
        return "BLANK"

    # Constants
    spec_frequencies, spec_segment_times, spec = spectrogram(sample, 200)
    samp = smooth_sample(sample, 10)
    smooth_spec_frequencies, smooth_spec_segment_times, smooth_spec = spectrogram(samp, 200)
    # np.savetxt('samples/latest_smoothed_sample', samp, delimiter=",")

    plt.pcolormesh(spec_segment_times, spec_frequencies, smooth_spec, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig('spec.png', bbox_inches='tight')

    plt.clf()
    plt.close()
    plt.pcolormesh(smooth_spec_segment_times, smooth_spec_frequencies, smooth_spec, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig('smooth_spec.png', bbox_inches='tight')

    #  go through the initial zeroes
    arr_index = 0
    smooth_array = samp.tolist()

    for x in smooth_array:
        if x <= 0 + leeway:
            arr_index += 1
        elif x > 0 + leeway:
            break
        else:   # nan
            arr_index += 1
            pass

    # Start counting
    tries = 0
    min_length = minimum_length
    while tries < 11 and min_length > 0:
        records.clear()
        buffers.clear()
        vib_counter(smooth_array, arr_index, min_length)

        if len(trueCode) > len(records) or len(trueCode) > (len(buffers)+1):
            print(records)
            print(buffers)
            print("Mismatch! Retrying with a more lenient min length. Retry number " + str(tries+1))
            min_length -= 1
        elif len(trueCode) < len(records) or len(trueCode) < (len(buffers)+1):
            print(records)
            print(buffers)
            print("Mismatch! Retrying with a stricter min length Retry number " + str(tries+1))
            min_length += 1
        else:
            break
        tries += 1

    if tries == 11 or min_length == 0:
        print("inaccurate recording for " + trueCode)
        exit()

    true_timings = []
    for c in trueCode:
        true_timings.append(reverse_lookup_table[c])

    # Begin analyzing
    record_ratios = [i / j for i, j in zip(true_timings, records)]
    rr_avg = sum(record_ratios) / len(record_ratios)
    computed_ratio = rr_avg

    # Experimental
    rr_median = statistics.median(record_ratios)
    for x in range(0, len(record_ratios)):
        val = record_ratios[x]
        if val > (rr_median * 3):
            record_ratios[x] = rr_median * 1.5
            computed_ratio = sum(record_ratios) / len(record_ratios)

    translated_records = [i * computed_ratio for i in records]
    rounded_timings = [int(round_to_nearest_ten(i)) for i in translated_records]

    hex_code = ""
    for i in rounded_timings:
        hex_code = hex_code + table_lookup(i)

    if log_results:
        print("------------------------------------")
        print("Original times:      " + str(true_timings))
        print("Calculated times:    " + str(rounded_timings))
        print("Original hex:        " + trueCode)
        print("Calculated hex:      " + hex_code)

        if len(trueCode) == len(hex_code):
            totalHexDistance = 0
            for i in range(0, len(hex_code)):
                diff = abs(int(hex_code[i], 16) - int(trueCode[i], 16))
                totalHexDistance += diff
            totalHexBits = len(hex_code) * 4    # Not always 4?
            percentCorrect = round(((totalHexBits - totalHexDistance) / totalHexBits) * 100, 4)
            print("Percent correct:     " + str(percentCorrect) + "%")

            if percentCorrect > 70:
                strPercent = str(percentCorrect)

                with open('./data_records/raw_data_'+trueCode + '_' + strPercent[0:strPercent.index('.')] + '.txt', 'w') as filehandle:
                    for num in sample:
                        filehandle.write('%d\n' % num)

                with open('./data_records/smooth_data_'+trueCode + '_' + strPercent[0:strPercent.index('.')] + '.txt', 'w') as filehandle:
                    for num in samp:
                        filehandle.write('%f\n' % num)



    end = time.time()
    print(str(end - start) + " seconds to process")
    return percentCorrect


@app.route("/phone_api", methods=["GET", "POST"])
def phone_api():
    global trueCode
    global fakeCode
    global buff
    global collectionTimeStamp
    global loginAttempt
    if request.method == "GET":
        if loginAttempt:
            loginAttempt = False
            return json.dumps({"data": trueCode})
        else:
            return json.dumps({"data": "null"})

    if request.method == "POST":
        print(request.form)
        collectionTimeStamp = request.form["collectionTimeStamp"]
        # fakeCode = request.form["fakeCode"]
        # buff = request.form["buffer"]
        return "Hello"
    return "Bad request"


@app.route("/welcome", methods=["GET"])
def welcome():
    if request.method == "GET":
        return render_template('success.html')


@app.route('/')
def index():
    print("Dispatching Secure Code...")
    if authenticated:
        return redirect(url_for("welcome"))
    return render_template('index.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    global loginAttempt
    global trueCode
    global collectionTimeStamp
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:6060')
    if request.method == "POST":
        collectionTimeStamp = ""
        trueCode = rand_key(hex_set, 8)
        loginAttempt = True
        print("key generated: " + trueCode)
    return response
    # return render_template("login.html", error=error)

# recives vibration and audio input
@app.route("/signal_api", methods=["GET", "POST"])
def signal_api():
    global collectionTimeStamp
    global authenticated
    global trueCode
    global fakeCode
    global buff
    if request.method == "GET":
        response = jsonify(
            {"collectionTimeStamp": collectionTimeStamp})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:6060')
        return response
    if request.method == "POST":
        audio=request.data.audio
        audioAccuracy=visualize(audio)
        data = np.array(json.loads(request.data.fullRecording.decode("utf-8"))["data"])
        # np.savetxt('raw_data', data, delimiter=",")
        codeAccuracy = process_signal(data)
        # time.sleep(5)
        if codeAccuracy >= 75.00 or audioAccuracy>=75:
            print("We're in")
            authenticated = True
            # return redirect(u0rl_for("welcome"))
        else:
            print("Not accepted")
            authenticated = False
        response = jsonify({"real": trueCode, "authenticated": authenticated})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:6060')
        # print(process_signal(data))
        # print(code)
        # print(incomingCode)
        collectionTimeStamp = ""
        return response

    return "Anything"


if __name__ == '__main__':
    app.run()


hex_table_setter()
