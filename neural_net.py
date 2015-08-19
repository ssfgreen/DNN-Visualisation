

class NeuralNet:

    # this is called when we do a post request
    def run(self,data):

        # this bit should be what runs
        print('running with {}'.format(data))

        # and returns this to...?
        return '123'

    # this is called when we do a results get request
    def result(self,timedate):

        # some middle stuff
        print('accessing data with timedate {}'.format(timedate))

        # this is the data passed back 
        return {'data':'hello'}