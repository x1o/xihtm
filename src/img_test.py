'''
Created on Oct 30, 2012

@author: xio
'''


import Image
from htm import *


def get_input_img(path, width=8):
    im = Image.open(path)
    im = im.convert('L')    # gray-scale; '1' for black and white
    inputs = [[int(s) for s in np.binary_repr(p, width)]
              for p in im.getdata()]
    return np.array(inputs).flatten()

def form_img(sdr):
    output_img = []
    for p in sdr:
        if p == 0:
            output_img.append('\xff')
        elif p == 1:
            output_img.append('\x00')
    return output_img

def save_img(sdr, dname, fname, t):
    side = int(np.sqrt(len(sdr)))
    Image.fromstring('L', (side, side), ''.join(sdr)).save(
            '%s/gen/%s_GEN_%s.%s' % (dname, fname.split('.')[0], t, fname.split('.')[1])
    )

def do_trial(region, trial, dname, fname, sdr_hist, stop_on_stable=False):
    logging.debug('\n\n=== trial: %s ===', trial)
    inputs = get_input_img(dname + '/' + fname, width=8)
#    logging.debug('inputs: %s\n' % inputs)
    region.connect_inputs(inputs)
    region.spatial_pooler()
    if stop_on_stable:
        if region.has_learned_input:
            logging.info('The region has learned the input, stopping')
            return 1

    if sdr_hist:
        if sdr_hist[-1] == region.sdr:
            diff_stat = '='
        else:
            diff_stat = '!'
    else:
        diff_stat = '?'
    logging.info('%s: |%s| %s= %s' % (trial, len(region.active_pds), diff_stat, trial-1))
    sdr_hist.append(region.sdr)
#    region.temporal_pooler()
    
    return 0
    
def main():
    logging.basicConfig(level=logging.INFO)

    # In Grok: 2048 columns, 12 neurons each.
    path = '/home/xio/Documents/HTM/img_tests/dice/1_16.bmp'
    input_size = len(get_input_img(path, width=8))
    dname = path.rsplit('/', 1)[0]
    n_layers = 1
    n_cells = 128
    
    def same_die():
        region = Region(n_layers, n_cells, input_size)
        num = 3
        fname = '%s_16.bmp' % num
        sdr_hist = []
        for epoch in range(1, 50):
            rval = do_trial(region, epoch, dname, fname, sdr_hist, True)
            save_img(form_img(region.sdr), dname, fname, epoch)
            if rval == 1:
                break
#            for pd in region.active_pds:
#                print pd
    
    def all_dice():
        region = Region(n_layers, n_cells, input_size)
        is_settled = False
        
        for epoch in range(100):
            if is_settled:
                break
            logging.info('=== Epoch %s ===' % epoch)
            sdr_hist = []
            for trial in range(1, 6):
                fname = '%s_16.bmp' % trial
                rval = do_trial(region, trial, dname, fname, sdr_hist, True)
                if rval == 1:
                    is_settled = True
                    break
        
        for num in range(1, 6):
            fname = '%s_16.bmp' % num
            inputs = get_input_img(dname + '/' + fname, width=8)
            region.connect_inputs(inputs)
            region.activate_winners()
            save_img(form_img(region.sdr), dname, fname, epoch)
        
        for pd in region.active_pds:
            for syn in pd.potential_synapses:
                if syn.permanence != 0 and syn.permanence != 1:
                    print pd
                    break
    
    def digits():
        path = '/home/xio/Documents/HTM/img_tests/digits/1.bmp'
        input_size = len(get_input_img(path, width=8))
        dname = path.rsplit('/', 1)[0]
        n_layers = 1
        n_cells = 256
        region = Region(n_layers, n_cells, input_size, init_inhib_radius=3)
        is_settled = False
        
        for epoch in range(50):
            if is_settled:
                break
            logging.info('=== Epoch %s ===' % epoch)
            sdr_hist = []
            for trial in range(1, 6):
                fname = '%s.bmp' % trial
                rval = do_trial(region, trial, dname, fname, sdr_hist, True)
                if rval == 1:
                    is_settled = True
                    break
        
        for num in range(1, 6):
            fname = '%s.bmp' % num
            inputs = get_input_img(dname + '/' + fname, width=8)
            region.connect_inputs(inputs)
            region.activate_winners()
            save_img(form_img(region.sdr), dname, fname, epoch)
        
        for pd in region.active_pds:
            for syn in pd.potential_synapses:
                if syn.permanence != 0 and syn.permanence != 1:
                    print pd
                    break

#    same_die()
    all_dice()
#    digits()


if __name__ == '__main__':
    main()