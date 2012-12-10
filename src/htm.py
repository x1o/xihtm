'''
Created on Oct 11, 2012

@author: xio
'''

import logging
import numpy as np
import random


STATES = {
        'inactive': 0,
        'active': 1,
        'predictive': 2,
        }
CONNECTED_PERM_THR = 0.2
MIN_OVERLAP = 2
INIT_BOOST = 1
INIT_INHIB_RADIUS = 3
DESIRED_LOCAL_ACTIVITY = 2
PERM_INCR = 0.2
PERM_DECR = 0.2
INIT_PERM = 0.3
N_LATERAL_CONNECTIONS = 2   # usually a dozen
PERC_INTERCONNECTED = 0.5   # percentage of cells laterally connected to a cell
ACTIVATION_THR = 2          # no. of valid synapses on a distal dendrite for it to be active
MIN_THR = 1                 # like ACTIVATION_THR, but during temporal pooler learning
N_NEW_SYNAPSES = 2
SHOW_CELL_DDS = True        # whether to show a cell's distal dendrites in reprs


class Hist(list):
    def __init__(self, *args, **kwargs):
        self.n = kwargs.get('n', 3)     # t = {0|-1|-2}; float('inf') for infinity
        try:
            del kwargs['n']
        except KeyError:
            pass
        super(Hist, self).__init__(*args, **kwargs)
    
    def __getitem__(self, t):
        try:
            rec = super(Hist, self).__getitem__(t-1)    # NB
        except IndexError as e:
#            logging.warn('%s: no record for time %s; returning for t=0' %
#                    (e.message, t))
            try:
                rec = super(Hist, self).__getitem__(-1)
            except IndexError as e:
                logging.critical('No record for the present, something is really wrong.')
                raise e
        return rec
    
    def __getslice__(self, i, j):
        return super(Hist, self).__getslice__(i-1, j-1)
        
    def append(self, obj):
        if len(self) >= self.n:
            del self[0]
        super(Hist, self).append(obj)
    
    def extend(self, iterable):
        for item in iterable:
            self.append(item)
    
    def insert(self, index, obj):
        raise NotImplementedError('Cannot edit history.')


class Cell(object):
    def __init__(self, idx, layer_idx, distal_dendrites=False):
        self.layer_idx = layer_idx
        self.idx = idx
        self.state_hist = Hist([])    # time-indexed history of states
        self.state_hist.append(STATES['inactive'])
        self.learning_states_hist = Hist([False])     # time-indexed
#        self.prox_dendrite    # adding later
        if distal_dendrites is False:
            self.distal_dendrites = []
        else:
            self.distal_dendrites = distal_dendrites
    
    def __repr__(self, show_cell_dds=SHOW_CELL_DDS):
        if self.state is False:
            state = '_'
        else:
            state = repr(self.state)
        if self.is_learning:
            state += 'L'
        retv = '%s/%s(%s)' % (self.layer_idx, self.idx, state)
        if SHOW_CELL_DDS:
            retv += ('~' + repr(self.distal_dendrites))
        return retv
    
    def get_state(self, t=0):
        return self.state_hist[t]
    
    @property
    def state(self):
        return self.get_state(t=0)
    
    @state.setter
    def state(self, val):
        self.state_hist.append(val)
    
    def set_prox_dendr(self, pd):
        self.prox_dendrite = pd
    
    def add_dist_dendr(self, dd):
        self.distal_dendrites.append(dd)
    
    @property
    def is_laterally_active(self):
        # TODO: make absolutely sure it is so.  is one segment enough?
        for dd in self.distal_dendrites:
            if dd.is_active():
                return True
        return False
    
    def get_active_dendrite(self, t=0):
        active_dds = [dd for dd in self.distal_dendrites if dd.was_active(t)]
        # FIXME: deal with zero active distal dendrites
        # XXX: the method is somewhat strange but follows Numenta's specifications
        if len(active_dds) == 0:
            msg = 'No segments are active'
            logging.error(msg)
            raise Exception(msg)
        elif len(active_dds) == 1:
            return active_dds[0]
        else:
            active_dds.sort(key=lambda dd: dd.n_active_synapses(t))
            for dd in active_dds:
                if dd.was_sequence_dendrite(t):
                    return dd
            return active_dds[-1]
    
    def was_learning(self, t=0):
        return self.learning_states_hist[t]
    
    @property
    def is_learning(self):
        return self.was_learning(t=0)
    
    @is_learning.setter
    def is_learning(self, val):
        self.learning_states_hist.append(val)
    
    @property
    def n_distal_dendrites(self):
        return len(self.distal_dendrites)
    
    def get_best_matching_dist_dendrite(self, t=0):
        active_dds = self.get_active_dist_dendrites(t, obey_act_threshold=False)
        if len(active_dds) == 0:
            return False
        else:
            active_dds.sort(
                    key=lambda dd: dd.get_n_active_synapses(t, obey_perm_threshold=False)
            )
            return active_dds[-1]
    
    def get_active_dist_dendrites(self, t, obey_act_threshold=True):
        return [dd for dd in self.distal_dendrites
                if dd.was_active(t, obey_act_threshold)]
    
    @property
    def active_dist_dendrites(self):
        return self.get_active_dist_dendrites(t=0)
    
    def discard_dd_updates(self):
        for dd in self.distal_dendrites:
            dd.discard_updates()
    
    def apply_dd_updates(self, positive_reinforcement):
        for dd in self.distal_dendrites:
            upd_syns = [upd[0] for upd in dd.update_list]
            if upd_syns:
                # flatten
                upd_syns = reduce(lambda u1, u2: u1.__add__(u2), upd_syns)
                if positive_reinforcement:
                    for syn in upd_syns:
                        syn.permanence += PERM_INCR
                    for syn in set(dd.potential_synapses) - set(upd_syns):
                        syn.permanence -= PERM_DECR
                else:
                    for syn in upd_syns:
                        syn.permanence -= PERM_DECR
            new_syn = set(upd_syns) - set(dd.potential_synapses)
            for syn in new_syn:
                syn.permanence = INIT_PERM
                dd.potential_synapses.append(syn)


class Synapse(object):
    def __init__(self, obj, inputs=False, permanence=False):
        # TODO:  abstract region inputs and cells
        #        or probably should we just have two types of synapses...
        self.obj = obj
        if inputs is not False:
            self.inputs = inputs
        if permanence is False:
            self._permanence = random.uniform(0, 1)
        else:
            self._permanence = permanence
    
    def __repr__(self):
        input_flag = ' '
        try:
            if self.gets_input:
                input_flag = '='
        except AttributeError:
            input_flag = '?'            
        valid_flag = ' '
        if self.is_valid:
            valid_flag = '+'        
        state = '%s%s:%s%.1f' % (input_flag, self.idx, valid_flag, self.permanence)
        return state
    
    @property
    def idx(self):
        if isinstance(self.obj, Cell):
            return self.obj.idx
        else:
            return self.obj
    
    @property
    def cell(self):
        if isinstance(self.obj, Cell):
            return self.obj
        else:
            raise AttributeError('This synapse is not connected to a cell.')
    
    @property
    def permanence(self):
        return self._permanence
    
    @permanence.setter
    def permanence(self, val):
        self._permanence = val
        self._permanence = min(1.0, max(0.0, self.permanence))     # constrain to [0, 1]

    @property
    def weight(self):
        if self.permanence >= CONNECTED_PERM_THR:
            return 1
        else:
            return 0
    
    @property
    def is_valid(self):
        return self.weight == 1
    
    @property
    def gets_input(self):
        return self.value > 0
    
    @property
    def value(self):
        if isinstance(self.obj, Cell):
            return self.cell.state
        else:
            return self.inputs[self.idx]
    
    def is_active(self, obey_perm_threshold=True):
        if isinstance(self.obj, Cell):
            if obey_perm_threshold:
                return self.is_valid & self.value == STATES['active']
            else:
                return self.value == STATES['active']
        else:
            return self.is_valid & self.gets_input


class Dendrite(object):
    pass


class ProximalDendrite(Dendrite):
    def __init__(self, idx, potential_syn_idxs, n_cells, init_boost, min_overlap):
        self.idx = idx
        self.n_cells = n_cells
        self.boost = init_boost
        self.min_overlap = min_overlap
        self.activation_hist = Hist([False], n=100)     # time-indexed
        self.overlaps_hist = Hist([], n=100)            # time-indexed
        # linearly decreasing permanence values with the peak at the center of
        # the input region
        idxs = sorted(potential_syn_idxs, key=lambda n: abs(self.idx - n))
        perms = np.linspace(CONNECTED_PERM_THR*2, 0, len(potential_syn_idxs))
        self.potential_synapses = [Synapse(idx, permanence=perm)
                                   for idx, perm in zip(idxs, perms)]
    
    def connect_inputs(self, inputs):
        for syn in self.potential_synapses:
            syn.inputs = inputs
    
    @property
    def active_synapses(self):
        return [s for s in self.potential_synapses if s.is_active()]
    
    def __repr__(self):
        if self.is_active:
            state = '*'
        else:
            state = ' '
#        return state + repr(self.connected_synapses)
        return state + repr(self.potential_synapses)
    
    @property
    def _real_overlap(self):
        return len([s for s in self.active_synapses])
    
    @property
    def overlap(self):
        real_overlap = self._real_overlap
        if real_overlap < self.min_overlap:
            self.overlaps_hist.append(False)
            return 0
        else:
            self.overlaps_hist.append(True)
            return real_overlap * self.boost
    
    def get_neighbour_idxs(self, radius):
        neigh = range(self.idx - radius, self.idx) + \
                range(self.idx + 1, self.idx + radius + 1)
        return [idx % self.n_cells for idx in neigh]
    
    @property
    def is_active(self):
        return self.activation_hist[0]

    def set_active(self):
        self.activation_hist.append(True)

    def set_inactive(self):
        self.activation_hist.append(False)
    
    def _get_rate(self, prop):
        return sum(prop) / float(len(prop))
    
    @property
    def firing_rate(self):
        return self._get_rate(self.activation_hist)
    
    @property
    def overlap_rate(self):
        return self._get_rate(self.overlaps_hist)
    
    def activate_cells(self, columns):
        for cell in columns[self.idx]:
            cell.state = STATES['active']
    
    @property
    def is_stable(self):
        for syn in self.potential_synapses:
            if syn.gets_input and syn.permanence != 1.0:
                return False
            elif (not syn.gets_input) and syn.permanence != 0:
                return False
        return True
    
    def get_best_matching_cell_dd(self, columns, t=0):
        column = columns[self.idx]
        cell_best_dd = [(cell, cell.get_best_matching_dist_dendrite(t))
                              for cell in column]
        if [pair[1] for pair in cell_best_dd].count(False) == len(cell_best_dd):
            # no best matching dendrites; no active cells -> no active synapses
            # -> no active dds?
            # return a cell with the smallest amount of dds 
            return (sorted(column, key=lambda cell: cell.n_distal_dendrites)[0],
                    False)
        else:
            # dd with largest n of active synapses
            cell_best_dd.sort(
                    key=lambda pair: pair[1] is not False and \
                                     pair[1].get_n_active_synapses(
                                             t, obey_perm_threshold=False
                                     )
            )
            return cell_best_dd[-1]


class DistalDendrite(Dendrite):
    def __init__(self, potential_synapses, is_sequence_dendrite=False):
        self.potential_synapses_hist = Hist([potential_synapses])     # time-indexed
        # does the segment predict ff input on the next time step?
        self.is_sequence_dendrite_hist = Hist([is_sequence_dendrite]) # time-indexed
        self.learning_states_hist = Hist([False])
        self.update_list = []
    
    def __repr__(self):
        state = ''
        if self.is_active():
            state += '*'
        if self.is_learning:
            state += 'L'
        if self.is_sequence_dendrite:
            state += 'S'
        return state + repr(self.potential_synapses)

    def get_potential_synapses(self, t=0):
        return self.potential_synapses_hist[t]
    
    @property
    def potential_synapses(self):
        return self.get_potential_synapses(t=0)
    
    @potential_synapses.setter
    def potential_synapses(self, val):
        self.potential_synapses_hist.append(val)
    
    def get_active_synapses(self, t=0, obey_perm_threshold=True):
        return [syn for syn in self.get_potential_synapses(t)
                if syn.is_active(obey_perm_threshold)]
    
    @property
    def active_synapses(self):
        return self.get_active_synapses(t=0)
    
    # no property for is_active -- need to supply an argument
    def is_active(self, obey_act_threshold=True):
        return self.was_active(t=0, obey_act_threshold=obey_act_threshold)
    
    def was_active(self, t=0, obey_act_threshold=True):
        n_active_synapses = len(self.get_active_synapses(t))
        if obey_act_threshold:
            return n_active_synapses > ACTIVATION_THR
        else:
            return n_active_synapses > MIN_THR
    
    def get_n_active_synapses(self, t=0, obey_perm_threshold=True):
        return len(self.get_active_synapses(t, obey_perm_threshold))
    
    def was_sequence_dendrite(self, t=0):
        return self.is_sequence_dendrite_hist[t]
    
    @property
    def is_sequence_dendrite(self):
        return self.was_sequence_dendrite(t=0)
    
    @is_sequence_dendrite.setter
    def is_sequence_dendrite(self, val):
        self.is_sequence_dendrite_hist.append(val)
    
    def was_learning(self, t=0):
        return self.learning_states_hist[t]
    
    @property
    def is_learning(self):
        return self.was_learning(t=0)
    
    @is_learning.setter
    def is_learning(self, val):
        self.learning_states_hist.append(val)
    
    def add_update(self, upd_synapses, is_sequence_segment=False):
        self.update_list.append((upd_synapses, is_sequence_segment))
    
    def discard_updates(self):
        del self.update_list[:]
    
    def get_update_active_synapses(self, t=0, cand_new_cells=False):
        active_synapses = [syn for syn in self.active_synapses
                if syn.cell.get_state(t) == STATES['active']]
        if cand_new_cells:
            n_new_syn = max(N_NEW_SYNAPSES - len(active_synapses), 0)
            for _ in range(n_new_syn):
                cell = random.choice(cand_new_cells)
                dd = random.choice(cell.distal_dendrites)
                syn = random.choice(dd.get_potential_synapses(t))
                active_synapses.append(syn)
        return active_synapses


class Region(object):
    def __init__(self, n_layers, n_cells, input_size,
                       init_inhib_radius=INIT_INHIB_RADIUS,
                       init_boost=INIT_BOOST,
                       n_dist_dendrites=N_LATERAL_CONNECTIONS,
                       perc_interconnected=PERC_INTERCONNECTED,
                       min_overlap=MIN_OVERLAP):
        logging.info('Creating a %s x %s region...' % (n_layers, n_cells))
        self.n_layers = n_layers    # per region
        self.n_cells = n_cells      # per layer
        self.region = np.array(
                [[Cell(c_idx, l_idx) for c_idx in range(n_cells)]
                 for l_idx in range(n_layers)]
        )
        self.inhib_radius = init_inhib_radius
        # add a shared proximal dendrite segment to each cell in a column
        input_idxs = set(range(input_size))
        n_syns = input_size / self.n_cells
        logging.info('Creating proximal dendrites with %s synapses each...' % n_syns)
        for col_idx, column in enumerate(self.columns):
#            potential_syn_idxs = list(set(
#                    [int(idx) % input_size for idx in np.random.normal(col_idx * step, 1, input_size)]
#            ))
            # connect to a random subset of inputs;
            # cover the whole input region, but w/o overlaps (contrary to
            #   Numenta's specs)
            potential_syn_idxs = random.sample(input_idxs, n_syns)
            input_idxs -= set(potential_syn_idxs)
            pd = ProximalDendrite(col_idx, potential_syn_idxs, self.n_cells,
                                  init_boost, min_overlap)
            for cell in column:
                cell.set_prox_dendr(pd)
        # randomly interconnect cells within each layer with distal dendrite
        # segments
        logging.info('Interconnecting cells with distal dendrites...')
        for i, layer in enumerate(self.layers):
            logging.info('  Level %s...' % i)
            for cell in layer:
                for _ in range(n_dist_dendrites):
                    cell_idxs = range(len(layer)); del cell_idxs[cell.idx]
                    n_potential_syn = int(perc_interconnected * len(cell_idxs))
                    potential_syn_idxs = set(
                            [random.choice(cell_idxs) for _ in range(n_potential_syn)]
                    )
                    potential_synapses = [Synapse(layer[idx]) for idx in potential_syn_idxs]
                    cell.add_dist_dendr(DistalDendrite(potential_synapses))
        logging.info('Region created.')
#        logging.debug('region: \n%s' % self)
    
    def __repr__(self):
        return repr(self.region)
    
    def __iter__(self):
        return iter(self.cells)
    
    @property
    def columns(self):
        return self.region.T
    
    @property
    def layers(self):
        return self.region
    
    @property
    def cells(self):
        return self.region.flatten()
    
    @property
    def prox_dendrites(self):
        return [cell.prox_dendrite for cell in self.layers[0]]
    
    @property
    def active_pds(self):
        return [pd for pd in self.prox_dendrites if pd.is_active]
    
    @property
    def sdr(self):
        return [int(pd.is_active) for pd in self.prox_dendrites]
    
    @property
    def overlaps(self):
        return [pd.overlap for pd in self.prox_dendrites]
    
    def connect_inputs(self, inputs):
        logging.debug('Connecting inputs...')
        for pd in self.prox_dendrites:
            pd.connect_inputs(inputs)
    
    def activate_winners(self, level=DESIRED_LOCAL_ACTIVITY):
        logging.debug('Activating winners...')
        overlaps = self.overlaps
#        logging.debug('overlaps: %s\n' % overlaps)
#        logging.debug('proximal dendrites:')
        for pd in self.prox_dendrites:
            neigh_idxs = pd.get_neighbour_idxs(self.inhib_radius)
            min_local_activity = sorted([overlaps[i] for i in neigh_idxs])[-level]
            if pd.overlap > min_local_activity:
                pd.set_active()
            else:
                pd.set_inactive()
#            logging.debug(pd)
    
    def sp_learn(self, perm_incr=PERM_INCR, perm_decr=PERM_DECR):
        logging.debug('Learning...')
        for pd in self.active_pds:
            for syn in pd.potential_synapses:
                if syn.gets_input:
                    syn.permanence += perm_incr
                else:
                    syn.permanence -= perm_decr
    
    def boost(self, connected_perm_thr=CONNECTED_PERM_THR):
        # TODO: test boosting
        logging.debug('Boosting...')
        for pd in self.prox_dendrites:
            neighbours = [self.prox_dendrites[idx]
                          for idx in pd.get_neighbour_idxs(self.inhib_radius)]
            min_duty_cycle = 0.01 * max([n.firing_rate for n in neighbours])
            cur_firing_rate = pd.firing_rate
            if cur_firing_rate >= min_duty_cycle:
                pd.boost = 1
            else:
                pd.boost += (min_duty_cycle - cur_firing_rate)
            
            if pd.overlap_rate < min_duty_cycle:
                for syn in pd.potential_synapses:
                    syn.permanence += 0.1 * connected_perm_thr
        # TODO: implement changing inhibition radius
#        self.inhib_radius = averageReceptiveFiledSize()

    @property
    def has_learned_input(self):
        if not self.active_pds:
            return False
        for pd in self.active_pds:
            if not pd.is_stable:
                return False
        return True
    
    def spatial_pooler(self, desired_local_activity=DESIRED_LOCAL_ACTIVITY,
                             perm_incr=PERM_INCR, perm_decr=PERM_DECR,
                             connected_perm_thr=CONNECTED_PERM_THR):
        logging.debug('--- spatial pooler ---')
    #    print region.columns[0][0].prox_dendr
    #    print region.columns[0][1].prox_dendr # same
    #    print region.columns[1][0].prox_dendr # different
        self.activate_winners(desired_local_activity)
        self.sp_learn(perm_incr, perm_decr)
    #    logging.debug('proximal dendrites after learning:')
    #    for pd in region.prox_dendrites:
    #        logging.debug(pd)
        self.boost(connected_perm_thr)
    #    logging.debug('region after sp: \n%s' % region)
        logging.debug('Active prox. dendrites:')
        for pd in self.active_pds:
            logging.debug(pd)
    
    def _get_cand_new_cells(self, cell, t=0):
        return [c for c in self.layers[cell.layer_idx] if c.was_learning(t=-1)]
    
    def infer_active_states(self):
        for pd in self.active_pds:
            is_predicted_input = False
            is_lc_chosen = False    # has the learning cell been chosen?
            for cell in self.columns[pd.idx]:
                if cell.get_state(t=-1) == STATES['predictive']:
                    try:
                        active_dendrite = cell.get_active_dendrite(t=-1)
                    except:
                        break
                    if active_dendrite.was_sequence_dendrite(t=-1):
                        is_predicted_input = True
                        cell.state = STATES['active']
                        if active_dendrite.was_learning(t=-1):
                            is_lc_chosen = True
                            cell.is_learning = True
            
            if not is_predicted_input:
                pd.activate_cells(self.columns)
            
            if not is_lc_chosen:
                # TODO: even if dd is False, add a new segment with no synapses;
                # how to organize updates, since dd doesn't exist?
                (cell, dd) = pd.get_best_matching_cell_dd(self.columns, t=-1)
                cell.is_learning = True
                # sequence update
                if dd is not False:
                    cand_new_cells = self._get_cand_new_cells(cell, t=-1)
                    upd_synapses = dd.get_update_active_synapses(t=-1, cand_new_cells=cand_new_cells)
                    dd.add_update(upd_synapses, is_sequence_segment=True)
    
    def infer_predictive_states(self):
        for cell in self.cells:
            # FIXME: make sure is_laterally_active's implementation is correct
            if cell.is_laterally_active:
                cell.state = STATES['predictive']
            
            for dd in cell.active_dist_dendrites:
                cand_new_cells = self._get_cand_new_cells(cell, t=-1)
                pred_dd = cell.get_best_matching_dist_dendrite(t=-1)
                pred_update = pred_dd.get_update_active_synapses(t=-1, cand_new_cells=cand_new_cells)
                active_update = dd.get_update_active_synapses(t=0, cand_new_cells=False)
                dd.add_update(active_update, is_sequence_segment=False)
                dd.add_update(pred_update, is_sequence_segment=False)
    
    def tp_learn(self):
        for cell in self.cells:
            if cell.is_learning:
                cell.apply_dd_updates(positive_reinforcement=True)
            elif cell.state != STATES['predictive'] \
                    and cell.get_state(t=-1) == STATES['predictive']:
                cell.apply_dd_updates(positive_reinforcement=False)
            cell.discard_dd_updates()
    
    def tp_output(self):
        output = []
        for cell in self.cells:
            if cell.state == STATES['active'] or cell.state == STATES['predictive']:
                output.append(1)
            else:
                output.append(0)
        return output

    def temporal_pooler(self):
        logging.debug('--- temporal pooler ---')
        logging.debug('region:\n%s' % self)
        self.infer_active_states()
        self.infer_predictive_states()
        self.tp_learn()
        logging.debug('region after tp: \n%s' % self)


def test():
    logging.basicConfig(level=logging.DEBUG)
    n_layers    = 3
    n_cells     = 16
    input_size  = 64
    perc_on     = 0.3
    n_trials    = 6
    output_hist = []
    region = Region(n_layers, n_cells, input_size)
    for t in range(n_trials):
        logging.debug('\n\n=== t: %s ===', t)
#        inputs = np.random.random_integers(0, 1, input_size)
#        inputs = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        # For testing purposes: create random inputs (the correlation between
        # inputs at $t_{0}$  and $t_{1}$  should be zero)
        inputs = [int(random.uniform(0, 1) > (1 - perc_on)) for _ in range(input_size)]
        logging.debug('inputs: %s' % inputs)
        region.connect_inputs(inputs)
        region.spatial_pooler()
        region.temporal_pooler()
        # Keep track of the region's outputs
        output_hist.append(region.tp_output())
    
    for o in output_hist:
        print '%s: %s' % (sum(o), o)

def main():
    pass


if __name__ == '__main__':
    test()