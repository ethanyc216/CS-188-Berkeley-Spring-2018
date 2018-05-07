import bisect

class EventQueue:
    """
      Implements the event queue for Pacman games.
      Currently uses a slow list implementation (not a heap) so that
      equality checking is easy.
    """
    def  __init__(self):
        self.sortedEvents = []

    def registerEventAtTime(self, event, time):
        assert isinstance(event, Event)
        entry = (time, event)
        index = bisect.bisect_right(self.sortedEvents, entry)
        self.sortedEvents.insert(index, entry)

    def peek(self):
        assert self.sortedEvents, "Error: Peek on an empty EventQueue"
        return self.sortedEvents[0]

    def pop(self):
        assert self.sortedEvents, "Error: Pop on an empty EventQueue"
        return self.sortedEvents.pop(0)

    def isEmpty(self):
        return len(self.sortedEvents) == 0

    def removeFirstEventSatisfying(self, f):
        index = 0
        l = len(self.sortedEvents)
        while index < l:
            _, event = self.sortedEvents[index]
            if f(event):
                return self.sortedEvents.pop(index)
            index += 1
        return None

    def getSortedTimesAndEvents(self):
        return self.sortedEvents

    def deepCopy(self):
        result = EventQueue()
        result.sortedEvents = [(t, e.deepCopy()) for t, e in self.sortedEvents]
        return result

    def __hash__(self):
        return hash(tuple(self.sortedEvents))

    def __eq__(self, other):
        return hasattr(other, 'sortedEvents') and \
            self.sortedEvents == other.sortedEvents

    def __str__(self):
        ansStr = "["
        for (t,e) in self.sortedEvents:
            if(e.isAgentMove):
                ansStr+= "(t = " + str(t) + ", " + str(e) + ")"
        return ansStr+"]"


class Event:
    """
    An abstract class for an Event.  All Events must have a trigger
    method which performs the actions of the Event.
    """
    nextId = 0
    def __init__(self, prevId=None):
        if prevId is None:
            self.eventId = Event.nextId
            Event.nextId += 1
        else:
            self.eventId = prevId

    def trigger(self, state):
        util.raiseNotDefined()

    def isAgentMove(self):
        return False

    def deepCopy(self):
        util.raiseNotDefined()

    def __eq__( self, other ):
        util.raiseNotDefined()

    def __hash__( self ):
        util.raiseNotDefined()

    def __lt__(self, other):
        return self.eventId < other.eventId
