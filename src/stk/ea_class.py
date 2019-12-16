import warnings
import os
import logging
import itertools as it
from rdkit import RDLogger
from os.path import join, basename
import stk
import psutil
import shutil
from importlib.machinery import SourceFileLoader

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)


# Define the formatter for logging messages.
try:
    f = '\n' + '='*os.get_terminal_size().columns + '\n\n'
except OSError:
    # When testing os.get_terminal_size() will fail because stdout is
    # not connected to a terminal.
    f = '\n' + '='*100 + '\n\n'
formatter = logging.Formatter(
    fmt=f+('%(asctime)s - %(levelname)s - %(name)s - %(message)s'),
    datefmt='%H:%M:%S'
)

# Define logging handlers.
errorhandler = logging.FileHandler(
    'output/scratch/errors.log',
    delay=True
)
errorhandler.setLevel(logging.ERROR)

streamhandler = logging.StreamHandler()

errorhandler.setFormatter(formatter)
streamhandler.setFormatter(formatter)

# Get the loggers.
rootlogger = logging.getLogger()
rootlogger.addHandler(errorhandler)
rootlogger.addHandler(streamhandler)

logger = logging.getLogger(__name__)


class EAHistory:
    """
    Deals with logging the EA's history.

    Attributes
    ----------
    fitness_calculator : :class:`.FitnessCalculator`
        The :class:`.FitnessCalculator` used to calculate fitness
        values.

    progress_dump : :class:`bool`
        Toggles dumping of :attr:`progress`.

    database_dump : :class:`bool`
        Toggles dumping of :attr:`dp_pop`.

    progress : :class:`.EAPopulation`
        A population where each subpopulation is a generation of the
        EA.

    db_pop : :class:`.EAPopulation` or :class:`NoneType`
        A population which holds every molecule made by the EA.

    dump_attrs : :class:`list` of :class:`str`, optional
        The names of attributes of the molecule to be added to
        the JSON.
    """

    def __init__(
        self,
        fitness_calculator,
        log_file,
        progress_dump,
        database_dump,
        dump_attrs,
    ):
        self.fitness_calculator = fitness_calculator
        self.log_file = log_file
        self.progress_dump = progress_dump
        self.database_dump = database_dump
        self.progress = stk.EAPopulation()
        self.db_pop = stk.EAPopulation()
        self.dump_attrs = dump_attrs

    def db(self, mols):
        """
        Adds `mols` to :attr:`db_pop`.

        Only molecules not already present are added.

        Parameters
        ----------
        mols : :class:`.EAPopulation`
            A group of molecules made by the EA.

        Returns
        -------
        None : :class:`NoneType`

        """

        if self.database_dump:
            self.db_pop.add_members(mols, duplicate_key=id)

    def dump(self):
        """
        Creates output files for the EA run.

        The following files are created:

            progress.log
                This file holds the progress of the EA in text form.
                Each generation is reprented by the names of the
                molecules and their key.

            progress.json
                A population dump file holding `progress`. Only made if
                :attr:`progress_dump` is ``True``.

            database.json
                A population dump file holding every molecule made by
                the EA. Only made if :attr:`db_pop` is not ``None``.

        """

        if self.log_file:
            with open('progress.log', 'w') as logfile:
                logfile.write(
                    '\n'.join(self.log_file_content(self.progress))
                )

        if self.progress_dump:
            self.progress.dump(
                'progress.json',
                include_attrs=self.dump_attrs,
            )
        if self.database_dump:
            self.db_pop.dump('database.json')

    @staticmethod
    def log_file_content(progress):
        for sp in progress.subpopulations:
            for mol in sp:
                yield f'{mol.id} {mol}'
            yield '\n'

    def log_pop(self, logger, pop):
        """
        Writes `pop` to `logger` at level ``INFO``.

        Parameters
        ----------
        logger : :class:`Logger`
            The logger object recording the EA.

        pop : :class:`.EAPopulation`
            A population which is to be added to the log.

        Returns
        -------
        None : :class:`NoneType`

        """

        if not logger.isEnabledFor(logging.INFO):
            return

        try:
            u = '-'*os.get_terminal_size().columns
        except OSError:
            # When testing os.get_terminal_size() will fail because
            # stdout is not connected to a terminal.
            u = '-'*100

        molecule = 'molecule'
        fitness = 'fitness'
        normalized_fitness = 'normalized fitness'
        rank = 'rank'

        fitness_values = pop.get_fitness_values()
        sorted_pop = sorted(
            pop,
            reverse=True,
            key=lambda m: fitness_values[m],
        )

        mols = '\n'.join(
            self.pop_log_content(sorted_pop, u, fitness_values)
        )
        s = (
            f'Population log:\n\n'
            f'{u}\n'
            f'{rank:<10}\t{molecule:<10}\t\t'
            f'{fitness:<40}\t{normalized_fitness}\n'
            f'{u}\n'
            f'{mols}'
        )

        with open('fitness.log', 'a') as f:
            f.write(s)

        logger.info(s)

    def pop_log_content(self, pop, underline, fitness_values):
        for i, mol in enumerate(pop, 1):
            fitness = self.fitness_calculator._cache[mol]
            yield (
                f'{i:<10}\t{mol}\t\t{fitness!r:<40}\t'
                f'{fitness_values[mol]}\n{underline}'
            )


class EvolutionaryAlgorithm:
    """
    Class representing the evolutionary algorithm.
    """

    def __init__(self):
        ...

    def init_from_file(self, path):
        # Load the input file as a module.
        loader = SourceFileLoader('input_file', path)
        input_file = loader.load_module()
        self.pop = input_file.population
        self.optimizer = input_file.optimizer
        self.fitness_calculator = input_file.fitness_calculator
        self.crosser = input_file.crosser
        self.mutator = input_file.mutator
        self.generation_selector = input_file.generation_selector
        self.mutation_selector = input_file.mutation_selector
        self.crossover_selector = input_file.crossover_selector
        self.terminator = input_file.terminator

        self.progress_dump_filename = join(
            '..',
            'pop_dumps',
            'progress.json'
        )
        self.database_dump_filename = join(
            '..',
            'pop_dumps',
            'progress.json'
        )

        self.fitness_normalizer = stk.NullFitnessNormalizer()
        if hasattr(input_file, 'fitness_normalizer'):
            self.fitness_normalizer = input_file.fitness_normalizer

        self.num_processes = psutil.cpu_count()
        if hasattr(input_file, 'num_processes'):
            self.num_processes = input_file.num_processes

        self.plotters = []
        if hasattr(input_file, 'plotters'):
            self.plotters = input_file.plotters

        self.log_file = True
        if hasattr(input_file, 'log_file'):
            self.log_file = input_file.log_file

        self.database_dump = True
        if hasattr(input_file, 'database_dump'):
            self.database_dump = input_file.database_dump

        self.progress_dump = True
        if hasattr(input_file, 'progress_dump'):
            self.progress_dump = input_file.progress_dump

        self.dump_attrs = None
        if hasattr(input_file, 'dump_attrs'):
            self.dump_attrs = input_file.dump_attrs

        self.debug_dumps = False
        if hasattr(input_file, 'debug_dumps'):
            self.debug_dumps = input_file.debug_dumps

        self.generation_dumps = True
        if hasattr(input_file, 'generation_dumps'):
            self.generation_dumps = input_file.generation_dumps

        self.tar_output = False
        if hasattr(input_file, 'tar_output'):
            self.tar_output = input_file.tar_output

        logging_level = logging.INFO
        if hasattr(input_file, 'logging_level'):
            logging_level = input_file.logging_level
        logging.getLogger('stk').setLevel(logging_level)
        logger.setLevel(logging_level)

        # EA should always use the cache.
        self.optimizer.set_cache_use(True)
        self.fitness_calculator.set_cache_use(True)
        self.crosser.set_cache_use(True)
        self.mutator.set_cache_use(True)

    def run_ea(self, filename):
        # Set up the directory structure.

        launch_dir = os.getcwd()

        # Running MacroModel optimizations sometimes leaves
        # applications open.This closes them. If this is not done,
        #  directories may not
        # be possible to move.
        stk.kill_macromodel()
        stk.archive_output()
        os.mkdir('output')
        os.chdir('output')
        root_dir = os.getcwd()

        if (self.database_dump
            or self.progress_dump
                or self.debug_dumps):
            os.mkdir('pop_dumps')

        # Copy the input script into the ``output`` folder.
        shutil.copyfile(filename, basename(filename))
        os.mkdir('scratch')
        os.chdir('scratch')
        open('errors.log', 'w').close()

        history = EAHistory(
            fitness_calculator=self.fitness_calculator,
            log_file=self.log_file,
            database_dump=self.database_dump,
            progress_dump=self.progress_dump,
            dump_attrs=self.dump_attrs,
        )
        progress = history.progress

        # 3. Run the EA.
        with self.pop.open_process_pool(self.num_processes):
            id_ = self.pop.set_mol_ids(0)
            logger.info('Optimizing the population.')
            # num_processes needs to be used explicitly in case
            # num_processes is set to 1.
            self.pop.optimize(
                self.optimizer,
                num_processes=self.num_processes,
            )

            logger.info(
                'Calculating the fitness of population members.'
            )
            self.pop.set_fitness_values_from_calculators(
                fitness_calculator=self.fitness_calculator,
                fitness_normalizer=self.fitness_normalizer,
                num_processes=self.num_processes,
            )

            history.log_pop(logger, self.pop)

            logger.info('Recording progress.')
            progress.add_subpopulation(self.pop)
            history.db(self.pop)

            gen = 0
            while not self.terminator.terminate(progress):
                gen += 1
                logger.info(f'Starting generation {gen}.')
                logger.debug(f'Population size is {len(self.pop)}.')

                logger.info('Creating offspring.')

                offspring_parents = self.crossover_selector.select(
                    self.pop,
                )
                offspring_batches = it.starmap(
                    self.crosser.cross,
                    offspring_parents,
                )
                # These need to be made here, outside of pop.extend
                # because otherwise the selector may select the things
                # which just got added, before their fitness got
                # calculated.
                offspring = list(
                    offspring
                    for offspring_batch in offspring_batches
                    for offspring in offspring_batch
                )

                logger.info('Creating mutants.')

                mutant_parents = self.mutation_selector.select(
                    self.pop,
                )
                mutants = list(it.starmap(
                    self.mutator.mutate,
                    mutant_parents,
                ))
                logger.info(
                    'Adding offspring and mutants into population.'
                )
                self.pop.direct_members.extend(it.chain(
                    offspring,
                    mutants,
                ))

                logger.debug(f'Population size is {len(self.pop)}.')

                logger.info('Removing duplicates, if any.')
                self.pop.remove_duplicates()

                logger.debug(f'Population size is {len(self.pop)}.')

                id_ = self.pop.set_mol_ids(id_)

                if self.debug_dumps:
                    self.pop.dump(join(
                        '..', 'pop_dumps', f'gen_{gen}_unselected.json'
                    ))

                logger.info('Optimizing the population.')
                self.pop.optimize(
                    self.optimizer,
                    num_processes=self.num_processes
                )

                logger.info(
                    'Calculating the fitness of population members.'
                )
                self.pop.set_fitness_values_from_calculators(
                    fitness_calculator=self.fitness_calculator,
                    fitness_normalizer=self.fitness_normalizer,
                    num_processes=self.num_processes,
                )

                history.log_pop(logger, self.pop)
                history.db(self.pop)

                logger.info(
                    'Selecting members of the next generation.'
                )
                self.pop.direct_members = list(
                    mol for mol, in self.generation_selector.select(
                        self.pop
                    )
                )

                history.log_pop(logger, self.pop)

                logger.info('Recording progress.')
                progress.add_subpopulation(self.pop)

                if self.generation_dumps:
                    self.pop.dump(
                        f'generation_{gen}.json',
                        include_attrs=self.dump_attrs,
                    )

                if self.debug_dumps:
                    progress.dump(self.progress_dump_filename)
                    history.db_pop.dump(self.database_dump_filename)

                stk.kill_macromodel()

                history.dump()

                for plotter in self.plotters:
                    plotter.plot(progress)

                os.chdir(root_dir)
                os.rename('scratch/errors.log', 'errors.log')

                self.pop.write('final_pop')
                os.chdir(launch_dir)
                if self.tar_output:
                    logger.info('Compressing output.')
                    stk.tar_output()
                stk.archive_output()
                logger.info('Successful exit.')
