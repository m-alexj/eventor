'''
Created on Apr 28, 2017

@author: judeaax
'''
import logging

logger = logging.getLogger("Eventor.index")


class Index(object):

    frozen = False

    UKN = 'UNKNOWN'

    def __init__(self, w2v):
        self.w2v = w2v

        def addDict(dict_name):
            self.__dict__["%s_index" % dict_name] = {}
            self.__dict__["inverse_%s_index" % dict_name] = {}

        addDict("word")
        self.addWord("n/a")

        addDict("entity")

        addDict("argument")
        self.addArgumentType("null")

        addDict("event")
        self.addEventType("No.Event")

        addDict("dependency")
        self.addDependency("none")
        self.addDependency("self")

        addDict("distance")
        self.addDistance("n/a")

        addDict("genre")

        addDict("pos")

        addDict("aux")

        addDict('feature')

        addDict('global_feature')

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['w2v']
        return state

    def __eq__(self, o):
        if isinstance(o, Index):
            for key, value in o.__dict__.items():
                if key not in self.__dict__:
                    return False
                if value != self.__dict__[key]:
                    return False
            return True
        return False

    def __ne__(self, o):
        return not self == o

    def addGlobalFeature(self, fid, force_add=False):
        return self.add(self.global_feature_index, self.inverse_global_feature_index, fid, force_add=force_add)

    def addFeature(self, fid, force_add=False):
        return self.add(self.feature_index, self.inverse_feature_index, fid, force_add=force_add)

    def addAux(self, aux):
        aix = self.add(self.aux_index, self.inverse_aux_index, aux, force_add=True)
        return aix

    def addPos(self, pos):
        pix = self.add(self.pos_index, self.inverse_pos_index, pos)
        return pix

    def addGenre(self, genre):
        gix = self.add(self.genre_index, self.inverse_genre_index, genre)
        return gix

    def addDistance(self, distance):
        didx = self.add(self.distance_index, self.inverse_distance_index, distance)
        return didx

    def addDependency(self, dependency):
        dpidx = self.add(self.dependency_index, self.inverse_dependency_index, dependency)
        return dpidx

    def addWord(self, word):
        return self.add(self.word_index, self.inverse_word_index, word)

    def getPos(self, pid):
        return self.inverse_pos_index[pid]

    def getWord(self, wid):
        if wid > 0:
            return self.inverse_word_index[wid]
        else:
            return self.UKN

    def getEntityType(self, eid):
        return self.inverse_entity_index[eid]

    def addEntityType(self, et):
        assert ' ' not in et
        return self.add(self.entity_index, self.inverse_entity_index, et)

    def addArgumentType(self, argument):
        return self.add(self.argument_index, self.inverse_argument_index,
                        argument)

    def addEventType(self, event):
        assert '.' in event
        return self.add(self.event_index, self.inverse_event_index, event)

    def getArgumentType(self, aid):
        return self.inverse_argument_index[aid]

    def getCleanDependency(self, did):
        dep = self.inverse_dependency_index[did]
        if dep.startswith('-'):
            dep = dep[1:]
        if dep.endswith('(-') or dep.endswith('-)'):
            dep = dep[:-2]
        return dep

    def getDependency(self, did):
        return (self.inverse_dependency_index[did]
                if did in self.inverse_dependency_index else None)

    def getEventType(self, eid):
        return self.inverse_event_index[eid]

    def getEventTypeLabels(self):
        return [self.getEventType(x) for x in range(len(self.event_index))]

    def getDistance(self, did):
        return self.inverse_distance_index[did]

    def add(self, index, inverse_index, value, force_add=False):
        if value not in index:
            if not force_add and self.frozen:
                return -1
            index[value] = len(index)
            inverse_index[index[value]] = value
        return index[value]

    def freeze(self):
        self.frozen = True
