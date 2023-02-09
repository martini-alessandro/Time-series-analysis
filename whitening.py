# specify the module that needs to be imported relative and its path
from importlib.machinery import SourceFileLoader

module_path = '/home/alessandro/Documenti/PhD/MESApaper/Maximum-Entropy-Spectrum/memspectrum.py'
module_name = 'memspectrum'
mem = SourceFileLoader(module_name,module_path).load_module() 

import numpy as np 
import matplotlib.pyplot as plt


class Data(object): 
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(y)
        self.duration = x[-1] - x[0]
        self.sampling_interval = self.duration / (self.length -1) 
        
      
        
    def compute_segments(self, segment_duration, lag,\
                                whitening_segments = 'overlap',\
                                whitening_duration=None):
        #Calls the segmentise class 
        s = Segments(self, segment_duration, lag)
        #Constrct the data segments
        self.filter_segments_x, self.filter_segments_y =\
            s.filter_segments_x, s.filter_segments_y
        self.whitening_segments_x, self.whitening_segments_y =\
            s.whitening_segments(method = whitening_segments,\
                                 duration = whitening_duration)

        return self.filter_segments_x, self.filter_segments_y,\
            self.whitening_segments_x, self.whitening_segments_y
    
    def whiten(self, method = 'lattice', **kwargs):
        w = Whitening(self)
        self.whitening_method = method.lower() 
        if self.whitening_method == 'lattice': 
            psd_method = kwargs.get('psd_method', 'MESA')
            self.ar_order = kwargs.get('ar_order', 300)
            self.whitened_segments = w.lattice(psd_method, ar_order)
        return None 
    
    def reunite_segments(self):
        
        if self.whitening_method == 'lattice':
            #Create two lists where the values of the whitened segments will be
            #stored. Since lattice method creates a lot of 0 values, they have
            #to be eliminated by getting rid of the first n (N = ar_order) elements.
            #Since this procedure changes the time grid, the same operation is 
            #performed over the whitening time segments 
            reunited_segments = []
            whitening_intervals = []
            for i, segment in enumerate(self.whitened_segments):
                whitening_interval = self.whitening_segments_x[i, self.ar_order:]
                non_zero_segments = segment[self.ar_order:]
                reunited_segments.append(non_zero_segments)
                whitening_intervals.append(whitening_interval)
            self.whitened_data = np.reshape(reunited_segments,\
                                        newshape = (np.size(reunited_segments),))
            self.whitening_interval = np.reshape(whitening_intervals,\
                                             newshape = np.size(whitening_intervals))
            
        elif self.whitening_method == 'subsequent':
            self.whitened_data = np.reshape(self.whitened_segments,\
                                            newshape = self.whitening_segments.size)
            self.whitened_interval = np.reshape(self.whitening_segments_x,\
                                                newshape = self.whitening_segments_x.size)
        return self.whitening_interval, self.whitened_data 
        
    
    
    
class Segments(object):
    
    def __init__(self, data, segment_duration, lag): 

        self.data = data
        self.segment_duration = segment_duration
        
        if lag < self.data.sampling_interval:
            raise ValueError('Lag duration must to be longer than data sampling\
                             interval. Sampling interval is {}, while chosen\
                                 value for lag is {}'.format(self.data.sampling_interval, lag))
        self.lag = lag
        self.filter_segments_x, self.filter_segments_y = self._compute_segments('filter')
        
    
    def whitening_segments(self, method='overlap', duration=None): 
            #IMPLEMENT NON OVERLAP OPTION (i.e. SUBSEQUENT or YOU CHOOSE WHAT)
            #IF SUBSQUENT ARE CHOSEN BEWARE LAG = WHITENING ITERVAL OTHERWISE WE 
            #WILL HAVE SOME NON-WHITENED PARTS 
            """
            Create the segments over which the whitening will be performed. The
            segments are defined as the overlap of two consequent data-segments
            over which the whitening filter is computed

            Parameters
            ----------
            segments : np.ndarra(N,2)
                an array containing the time-boundaries of each subsegment of data.
                The proper data segments can be generated via the method 
                "self._create_data_segments()"

            Returns
            -------
            whitening_segments : TYPE
                Buondaries of the segments over which the data will be whitened..

            """
            if method.lower() == 'overlap': 
               self.whitening_segments_x, self.whitening_segments_y = \
                   self._compute_segments('overlap')
            elif method.lower() == 'subsequent':    
                self.whitening_segments_x, self.whitening_segments_y = \
                    self._compute_segments('subsequent', duration)
            return self.whitening_segments_x, self.whitening_segments_y
        
    def _compute_segments(self, segment_type, white_segment_duration = None):

        #Transforms the time length in array length
        segment_length = int(self.segment_duration / self.data.sampling_interval)
        lag_length = int(self.lag / self.data.sampling_interval)
        #Computes the segments based on chosen type
        if segment_type.lower() == 'filter': 
            segments_x, segments_y = self._filter_segments(segment_length, lag_length)
        elif segment_type.lower() == 'overlap':
            segments_x, segments_y = self._overlap_segments(segment_length, lag_length) 
        elif segment_type.lower() == 'subsequent':
            segments_x,segments_y = self._subsequent_segments(segment_length, lag_length,\
                                                              white_segment_duration)
        else:
            raise ValueError('Segment type has to be "Filter", "Overlap" or\
                             "Subsequent"')
          
        return segments_x, segments_y
    
    def _filter_segments(self, segment_length, lag_length): 
        i_max = int((self.data.length - segment_length) / lag_length)
        segments_x, segments_y = [], []
        for i in range(i_max): 
                xsegment = self.data.x[i*lag_length:i*lag_length+segment_length]
                ysegment = self.data.y[i*lag_length:i*lag_length+segment_length]
                segments_x.append(xsegment)
                segments_y.append(ysegment)
        return np.array(segments_x), np.array(segments_y)
     
    def _overlap_segments(self, segment_length, lag_length): 
        i_max = int((self.data.length - segment_length) / lag_length)
        segments_x, segments_y = [], []
        for i in range(i_max): 
            xsegment = self.data.x[(i+1)*lag_length:i*lag_length + segment_length]
            ysegment = self.data.y[(i+1)*lag_length:i*lag_length + segment_length]
            segments_x.append(xsegment)
            segments_y.append(ysegment)
        return np.array(segments_x), np.array(segments_y)

    def _subsequent_segments(self, segment_length, lag_length, whitening_duration):
        white_segment_length = int(whitening_duration/self.data.sampling_interval)
        i_max = int((self.data.length - segment_length - white_segment_length)\
                    / lag_length)
        segments_x, segments_y = [], []
        for i in range(i_max):
            xsegment = self.data.x[i*lag_length+segment_length:\
                                   i*lag_length+segment_length+white_segment_length]
            ysegment = self.data.y[i*lag_length+segment_length:\
                                   i*lag_length+segment_length+white_segment_length]
            segments_x.append(xsegment)
            segments_y.append(ysegment)
        return np.array(segments_x), np.array(segments_y)
 
    
class Whitening(object):
    
    def __init__(self, data):
     
        self.data = data
        try:
            self.data.filter_segments_x
        except AttributeError: 
            raise AttributeError('The data must be segmentised using\
                  data.create_segments() method before whitening')
                  
    
    def MESA_psd_division(self, optimisation_method, m = None):
        #Compute PSD over filter segments
        M = mem.MESA() 
        dt = self.data.sampling_interval
        whitened_segments = []
        for i, data in enumerate(self.data.whitening_segments_y):
            M.solve(self.data.filter_segments_y[i], optimisation_method = optimisation_method)
            f_data = np.fft.fft(data)
            frequencies = np.fft.fftfreq(len(f_data), dt)
            print(frequencies)
            spectrum = M.spectrum(dt, frequencies)
            if i ==1: self.iszero = (np.any(spectrum == 0)).sum() ###On of this is zero! 
            f_data /= spectrum
            whitened = np.fft.ifft(f_data)
            whitened_segments.append(whitened)
        self.whitened_segments = np.array(whitened_segments)
        return self.whitened_segments
    
    def lattice(self, method = 'MESA', ar_order = 'FPE'):
        """
        
        Notice that lattice automatically set to zero the first p-elements of the array. 
        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'MESA'.
        ar_order : TYPE, optional
            DESCRIPTION. The default is 'FPE'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        #Takes segments for filter and compute the filter at given AR order
        self.reflection_coefficients, self.ar_orders =\
            self._compute_reflection_coefficients(method, ar_order)
            
        #Implements lattice whitening filter on  whitening segments
        whitened_segments = []
        for i, data in enumerate(self.data.whitening_segments_y):
            forward_error, backward_error=\
                self._prediction_errors(data, self.reflection_coefficients[i],\
                                        self.ar_orders[i])
            whitened_segments.append(forward_error)
        self.whitened_segments = np.array(whitened_segments)
        return self.whitened_segments 
    
    def _compute_reflection_coefficients(self, method = 'MESA', ar_order = 'FPE'):
        reflection_coefficients = [] 
        M = mem.MESA() 
        ar_orders = []
        for segment in self.data.filter_segments_y: 
            
            if type(ar_order) == int:
                P, ak, opt, reflection_coeff =\
                    M.solve(segment, m = ar_order, optimisation_method = 'Fixed')
            
            elif type(ar_order) == str:
                P, ak, opt, reflection_coeff =\
                    M.solve(segment, optimisation_method = ar_order)
                
            reflection_coefficients.append(reflection_coeff)
            ar_orders.append(M.get_p())
        
            
        return np.array(reflection_coefficients), np.array(ar_orders)
    

    def _prediction_errors(self, data, reflection_coefficients, ar_order):
    
        FPE = np.zeros((ar_order + 1, data.size))
        BPE = np.zeros((ar_order + 1, data.size))
        FPE[0], BPE[0] = data, data
        for i in range(ar_order): 
            FPE[i+1,i+1:] = FPE[i, i+1:] + reflection_coefficients[i] * BPE[i,i:-1]
            BPE[i+1,i+1:] = BPE[i,i:-1] + reflection_coefficients[i] * FPE[i,i+1:]
        return FPE[-1], BPE[-1]
   

if __name__ == '__main__':
    import_ = False
    if import_: 
        data = np.loadtxt('/home/alessandro/Documenti/PhD/MESApaper/Maximum-Entropy-Spectrum/examples/On_source_toymodel/Noise_series/noise_realisations.txt')
        dt = 1 / 4096
    compute_FPE = False
    
    segments_creation = True
    if segments_creation:
        data_x = np.linspace(0,30,10000)
        data_y = np.sin(np.pi * data_x)
        data_y += np.random.normal(0, 1, 10000)
        D = Data(data_x, data_y)
        #To compute segments for lattice filter, remember that some values#
        #are put to 0. So lag = (duration - seconds_lost) / 2. And seconds
        #los = ar_order * sampling_interval
        ar_order = 300
        duration = 5
        lag = (duration - ar_order * D.sampling_interval) / 2
        D.compute_segments(duration, lag)
        D.whiten(ar_order = 300)
        D.reunite_segments()
        
    
    
    
    
    
