import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import stats
# Part Z: Bipolar 
class BPStochasticComputing:

    @classmethod
    def to_unip(cls, bip):
        return 0.5 * (bip+1) 
    
    @classmethod
    def to_bip(cls, unip):
        return 2(unip-0.5)
    
    @classmethod
    def bip_add(cls, bitstream, bitstream2):
        assert(len(bitstream) == len(bitstream2))
        newbitstream = []
        for i in range(len(bitstream)):
            X = stats.binom(1, 0.5)
            if X == 0: 
                newbitstream.append(bitstream[i])
            else:
                newbitstream.append(bitstream2[i])
        return newbitstream


    @classmethod
    def bip_mul(cls, bitstream, bitstream2):
        assert(len(bitstream) == len(bitstream2))
        newbitstream = []
        for i in range(len(bitstream)):
           newbitstream.append(bitstream[i] & bitstream2[i] + (1-bitstream[i]) & (1-bitstream2[i]))
        return newbitstream

class PosStochasticComputing:
    APPLY_FLIPS = False
    APPLY_SHIFTS = False

    @classmethod
    def apply_bitshift(cls, bitstream):
        if not PosStochasticComputing.APPLY_SHIFTS:
            return bitstream
        for i in range(len(bitstream)-1):
            if stats.binom(1, 0.0001) == 1:
                for j in (i, len(bitstream)-len(bitstream)):
                    bitstream[len(bitstream)-1-j] = bitstream[len(bitstream)-2-j] 


    @classmethod
    def apply_bitflip(self, bitstream):
        if not PosStochasticComputing.APPLY_FLIPS:
            return bitstream
        for i in range(len(bitstream)):
            if stats.binom(1, 0.0001) == 1:
                bitstream[i] = ~bitstream[i]
        raise Exception("apply the to the bitstream with probability 0.0001")



    @classmethod
    def to_stoch(cls, prob, nbits):
        assert(prob <= 1.0 and prob >= 0.0)
        X = np.random.binomial(1, prob, nbits)
        return X
            

    @classmethod
    def stoch_add(cls, bitstream, bitstream2):
        assert(len(bitstream) == len(bitstream2))
        newbitstream = []
        for i in range(len(bitstream)):
            X = stats.binom(1, 0.5)
            if X == 0: 
                newbitstream.append(bitstream[i])
            else:
                newbitstream.append(bitstream2[i])
        return newbitstream


    @classmethod
    def stoch_mul(cls, bitstream, bitstream2):
        assert(len(bitstream) == len(bitstream2))
        newbitstream = []
        for i in range(len(bitstream)):
           newbitstream.append(bitstream[i] & bitstream2[i])
        return newbitstream
    
    @classmethod
    def from_stoch(cls, result):
        x = 0
        for i in range(len(result)):
            if result[i] == 1:
                x = x + 1
        return x/len(result)

class StochasticComputingStaticAnalysis:

    def __init__(self):
        self.precisions = []

    def req_length(self, smallest_value):
        return int(1 / smallest_value)

    def stoch_var(self, prec):
        self.precisions.append(prec)
        return prec

    def stoch_add(self, prec1, prec2):
        result_prec = max(prec1, prec2)  
        self.precisions.append(result_prec)
        return result_prec


    def stoch_mul(self, prec1, prec2):
        result_prec = prec1 * prec2  
        self.precisions.append(result_prec)
        return result_prec

    def get_size(self):
        smallest_precision = min(self.precisions)
        return self.req_length(smallest_precision)



# run a stochastic computation for ntrials trials
def run_stochastic_computation(lambd, ntrials, visualize=True, summary=True):
    results = []
    reference_value, _ = lambd()
    for i in range(ntrials):
        _,result = lambd()
        results.append(result)

    if visualize:
        nbins = math.floor(np.sqrt(ntrials))
        plt.hist(results,bins=nbins)
        plt.axvline(x=reference_value, color="red")
        plt.show()
    if summary:
        print("ref=%f" % (reference_value))
        print("mean=%f" % np.mean(results))
        print("std=%f" % np.std(results))




def PART_A_example_computation(bitstream_len):
    # expression: 1/2*(0.8 * 0.4 + 0.6)
    reference_value = 1/2*(0.8 * 0.4 + 0.6)
    w = PosStochasticComputing.to_stoch(0.8, bitstream_len)
    x = PosStochasticComputing.to_stoch(0.4, bitstream_len)
    y = PosStochasticComputing.to_stoch(0.6, bitstream_len)
    tmp = PosStochasticComputing.stoch_mul(x, w)
    result = PosStochasticComputing.stoch_add(tmp, y)
    return reference_value, PosStochasticComputing.from_stoch(result)


def PART_Y_analyze_wxb_function(precs):
    # 1/2*(w*x + b)
    analysis = StochasticComputingStaticAnalysis()
    w_prec = analysis.stoch_var(precs["w"])
    x_prec = analysis.stoch_var(precs["x"])
    b_prec = analysis.stoch_var(precs["b"])
    res_prec = analysis.stoch_mul(w_prec, x_prec)
    analysis.stoch_add(res_prec, b_prec)
    N = analysis.get_size()
    print("best size: %d" % N)
    return N

def PART_Y_execute_wxb_function(values, N):
    # expression: 1/2*(w*x + b)
    w = values["w"]
    x = values["x"]
    b = values["b"]
    reference_value = 1/2*(w*x + b)
    w = PosStochasticComputing.to_stoch(w, N)
    x = PosStochasticComputing.to_stoch(x, N)
    b = PosStochasticComputing.to_stoch(b, N)
    tmp = PosStochasticComputing.stoch_mul(x, w)
    result = PosStochasticComputing.stoch_add(tmp, b)
    return reference_value, PosStochasticComputing.from_stoch(result)


def PART_Y_test_analysis():
    precs = {"x": 0.1, "b":0.1, "w":0.01}
    # apply the static analysis to the w*x+b expression, where the precision of x and b is 0.1 and
    # the precision of w is 0.01
    N_optimal = PART_Y_analyze_wxb_function(precs)
    print("best size: %d" % N_optimal)

    variables = {}
    for _ in range(10):
        variables["x"] = round(np.random.uniform(),1)
        variables["w"] = round(np.random.uniform(),2)
        variables["b"] = round(np.random.uniform(),1)
        print(variables)
        run_stochastic_computation(lambda : PART_Y_execute_wxb_function(variables,N_optimal), ntrials=10000, visualize=True)
        print("")


def PART_Z_execute_rng_efficient_computation(value,N,save_rngs=True):
    # expression: 1/2*(x*x+x)
    xv = value
    reference_value = 1/2*(xv*xv + xv)
    if save_rngs:
        x = PosStochasticComputing.to_stoch(xv, N)
        x2 = x
        x3 = x
    else:
        x = PosStochasticComputing.to_stoch(xv, N)
        x2 = PosStochasticComputing.to_stoch(xv, N)
        x3 = PosStochasticComputing.to_stoch(xv, N)

    tmp = PosStochasticComputing.stoch_mul(x, x2)
    result = PosStochasticComputing.stoch_add(tmp,x3)
    return reference_value, PosStochasticComputing.from_stoch(result)


"""
print("---- part a: effect of length on stochastic computation ---")
ntrials = 10000
run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=10), ntrials)
run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=100), ntrials)
run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)


# Part X, introduce non-idealities
PosStochasticComputing.APPLY_FLIPS = True
PosStochasticComputing.APPLY_SHIFTS =False
print("---- part x: effect of bit flips ---")
run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
PosStochasticComputing.APPLY_FLIPS = False
PosStochasticComputing.APPLY_SHIFTS = True
print("---- part x: effect of bit shifts ---")
run_stochastic_computation(lambda : PART_A_example_computation(bitstream_len=1000), ntrials)
PosStochasticComputing.APPLY_FLIPS = False
PosStochasticComputing.APPLY_SHIFTS =False
"""

# Part Y, apply static analysis
print("---- part y: apply static analysis ---")
PART_Y_test_analysis()

# Part Z, resource efficent rng generation
print("---- part z: one-rng optimization ---")
for _ in range(5):
    v = round(np.random.uniform(),1)
    print(f"x = {v}")
    print("running with save_rngs disabled")
    run_stochastic_computation(lambda : PART_Z_execute_rng_efficient_computation(value=v, N=1000, save_rngs=False), ntrials)
    print("running with save_rngs enabled")
    run_stochastic_computation(lambda : PART_Z_execute_rng_efficient_computation(value=v, N=1000, save_rngs=True), ntrials)
