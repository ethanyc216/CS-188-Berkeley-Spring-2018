import bayesNet
from copy import deepcopy


def powersBayesNet():
    powerVars = ['laser', 'capacity', 'invisibility', 
                     'respawn', 'speed']
    sumVars = ['sum0', 'sum1', 'sum2', 'sum3']
    variablesList = powerVars + sumVars

    variableDomainsDict = {} 
    variableDomainsDict['laser'] = range(3)
    variableDomainsDict['speed'] = range(3)
    variableDomainsDict['capacity'] = range(3)
    variableDomainsDict['invisibility'] = range(3)
    variableDomainsDict['respawn'] = range(3)

    variableDomainsDict['sum0'] = range(6)
    variableDomainsDict['sum1'] = range(9)
    variableDomainsDict['sum2'] = range(9)
    variableDomainsDict['sum3'] = range(6)


    """
    edgeTuplesList = [('laser', 'sum0'), ('laser', 'sum2'),
                      ('capacity', 'sum0'), ('capacity', 'sum1'),
                      ('invisibility', 'sum0'), ('invisibility', 'sum1'),
                      ('respawn', 'sum1'), ('respawn', 'sum2'),
                      ('speed', 'sum1'), ('speed', 'sum2'),
                      #('speed', 'sum1')  ('speed', 'sum3')
                      ]
    """
    edgeTuplesList = [('laser', 'sum1'), ('laser', 'sum2'),
                      ('capacity', 'sum0'), ('capacity', 'sum2'),
                      ('invisibility', 'sum1'), ('invisibility', 'sum3'),
                      ('respawn', 'sum0'), ('respawn', 'sum1'),
                      ('speed', 'sum2'), ('speed', 'sum3'),
                      #('speed', 'sum1') # ('speed', 'sum3')
                      ]


    net = bayesNet.constructEmptyBayesNet(variablesList, edgeTuplesList, variableDomainsDict)
    
    def sumFactor(sumVar, inputVars):
        newFactor = bayesNet.Factor([sumVar], inputVars, variableDomainsDict)
        # only need to set the ones
        for assignmentDict in newFactor.getAllPossibleAssignmentDicts():
            inputSum = sum([int(assignmentDict[inputVar]) for inputVar in inputVars])
            if inputSum == int(assignmentDict[sumVar]):
                newFactor.setProbability(assignmentDict, 1.0)
        
        return newFactor

    def powerFactor(powerVar):
        newFactor = bayesNet.Factor([powerVar], [], variableDomainsDict)
        numChoices = len(variableDomainsDict[powerVar])
        for assignmentDict in newFactor.getAllPossibleAssignmentDicts():
            newFactor.setProbability(assignmentDict, 1.0 / numChoices)

        return newFactor

    inEdges = net.inEdges()
    for sumVar in sumVars:
        inputVars = inEdges[sumVar]
        net.setCPT(sumVar, sumFactor(sumVar, inputVars))

    for powerVar in powerVars:
        net.setCPT(powerVar, powerFactor(powerVar))

    return net


def computeMarginals(sumEvidence):
    net = powersBayesNet()


    powerVars = ['laser', 'capacity', 'invisibility', 
                     'respawn', 'speed']
    marginals = []
    for powerVar in powerVars:
        import inference
        marginal = inference.inferenceByVariableElimination(net, [powerVar], sumEvidence, None)
        # print "powerVar: " + powerVar 
        # print marginal
        marginals.append(marginal)
    return marginals


def computeEvidence(powerAssignments):
    net = powersBayesNet()
    inEdges = net.inEdges()
    powerVars = ['laser', 'capacity', 'invisibility', 
                     'respawn', 'speed']
    powerAssignmentsCopy = deepcopy(powerAssignments)
    for powerVar in powerVars:
        if powerVar not in powerAssignments:
            powerAssignmentsCopy[powerVar] = 0
    sumVars = ['sum0', 'sum1', 'sum2', 'sum3']
    sumEvidence = {}
    for sumVar in sumVars:
        inputVars = inEdges[sumVar]
        inputSum = sum([int(powerAssignmentsCopy[inputVar]) for inputVar in inputVars])
        sumEvidence[sumVar] = inputSum

    return sumEvidence


if __name__ == "__main__":
    powerVars = ['laser', 'capacity', 'invisibility', 
                     'respawn', 'speed']

    powerValues = [1, 0, 0, 1, 1]

    powerAssignments = dict(zip(powerVars, powerValues))
    computeMarginals(computeEvidence(powerAssignments))




