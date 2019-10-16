import stk


def test_any_terminator():
    pop1 = stk.Population(*(stk.Population() for i in range(5)))
    pop2 = stk.Population(*(stk.Population() for i in range(10)))
    pop3 = stk.Population(*(stk.Population() for i in range(20)))

    terminator = stk.AnyTerminator(
        stk.NumGenerations(7),
        stk.NumGenerations(15)
    )
    assert not terminator.terminate(pop1)
    assert terminator.terminate(pop2)
    assert terminator.terminate(pop3)


def test_all_terminators():
    pop1 = stk.Population(*(stk.Population() for i in range(5)))
    pop2 = stk.Population(*(stk.Population() for i in range(10)))
    pop3 = stk.Population(*(stk.Population() for i in range(20)))

    terminator = stk.AllTerminators(
        stk.NumGenerations(7),
        stk.NumGenerations(15)
    )
    assert not terminator.terminate(pop1)
    assert not terminator.terminate(pop2)
    assert terminator.terminate(pop3)


def test_num_generations():
    pop1 = stk.Population(*(stk.Population() for i in range(5)))
    pop2 = stk.Population(*(stk.Population() for i in range(10)))
    pop3 = stk.Population(*(stk.Population() for i in range(15)))

    terminator = stk.NumGenerations(10)

    assert not terminator.terminate(pop1)
    assert terminator.terminate(pop2)
    assert terminator.terminate(pop3)


def test_molecule_present(amine2):
    pop1 = stk.Population(stk.Population())
    pop2 = stk.Population(stk.Population(amine2))

    terminator = stk.MoleculePresent(amine2)

    assert not terminator.terminate(pop1)
    assert terminator.terminate(pop2)


def test_fitness_plateau():
    bbs = [
        stk.BuildingBlock.__new__(stk.BuildingBlock)
        for i in range(5*10)
    ]
    pop = stk.EAPopulation(
        *(stk.EAPopulation(*bbs[i:i+5]) for i in range(0, len(bbs), 5))
    )

    fitness_values = {}
    for i, mol in enumerate(pop):
        mol._identity_key = i
        mol.__class__.__str__ = lambda mol: str(mol._identity_key)
        fitness_values.update(
            {mol: i}
        )

    # Setting subpop fitness.
    for subpop in pop.subpopulations:
        subpop_fitness = {}
        for mol in subpop:
            subpop_fitness.update(
                {mol: fitness_values[mol]}
            )
        subpop.set_fitness_values_from_dict(subpop_fitness)

    terminator = stk.FitnessPlateau(2)

    assert not terminator.terminate(pop)

    fitness_values = {}

    # Set all fitness values to 2.
    for subpop in pop.subpopulations:
        subpop_fitness = {}
        for mol in subpop:
            subpop_fitness.update(
                {mol: 2}
            )
        subpop.set_fitness_values_from_dict(subpop_fitness)

    assert not terminator.terminate(pop)

    pop2 = stk.EAPopulation(*pop)
    pop3 = stk.EAPopulation(pop2, pop2)

    # Adding fitness to subpopulations.
    for subpop in pop3.subpopulations:
        subpop_fitness = {}
        for i, mol in enumerate(subpop):
            subpop_fitness.update(
                {mol: i}
            )
        subpop.set_fitness_values_from_dict(subpop_fitness)

    assert terminator.terminate(pop3)
