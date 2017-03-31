"""
Module with utilities to operate on list of SLCs;
especially useful in combination with snakemake to
select a certain range of dates, compute all interferograms etc

"""

import csv as _csv
import datetime as _dt
import itertools as _iter
import operator as _op
import pickle

import numpy as _np

import networkx as _nx


def try_format(dt, fmt):
    try:
        dt.hour
    except AttributeError:
        return _dt.datetime.strptime(dt, fmt)
    else:
        return dt


def format_deco(fun):
    def wrapper(self, *args, **kwargs):
        return self.format_date(fun(self,*args, **kwargs))
    return wrapper


class ListOfDates():
    """
    Helper class to represent a list of dates in a "Fuzzy" way.
    It permits 'soft' slicing: by slicing the object
    using a range of dates that may not be contained in the list, the closest date are chosen to slice the list of dates
    """

    @classmethod
    def fromfile(cls, fname,delimiter=',' ,**kwargs):
        """
        Classmethod to create a :obj:`ListOfDates` object from a csv file
        Parameters
        ----------
        fname : :obj:`str`
            The filename
        delimiter : optional, :obj:`str`
            The csv delimiter
        Returns
        -------

        """
        with open(fname, 'r') as f:  # the object contains a list of valid slcs
            # reader = _csv.reader(f, delimiter=delimiter)
            dat = [lin.strip() for lin in f.readlines()]
            all_dates = ListOfDates(dat, **kwargs)
        return all_dates

    @classmethod
    def fromregexp(cls, list_of_names,re, **kwargs):
        """
        Classmethod to create a :obj:`ListOfDates` object from a list
        using a regex to extract the date

        Parameters
        ----------
        re
        kwargs

        Returns
        -------

        """
        dt = [re.search(dt_string).group(0) for dt_string in list_of_names]
        all_dates = ListOfDates(dt, **kwargs)
        return all_dates



    def __init__(self, dates, date_format="%Y%m%d_%H%M%S", format_ouput=True):
        """
        Initializes the ListOfDates object

        Parameters
        ----------
        dates : :obj:`iterable` of :obj:`str` or :obj:`datetime`
            List of dates either as a list of strings or of datetime objects
        date_format : :obj:`str`
            String to describe the parsing format, as used by datetime.datetime.strptime

        """
        formatted_dates = []
        for d in dates:
            try:
               formatted_dates.append( _dt.datetime.strptime(d, date_format))
            except ValueError:
                continue
        self.dates = _np.array(sorted(formatted_dates))
        self.format = date_format
        self.format_output = format_ouput

    def __len__(self):
        return self.dates.__len__()

    def __getitem__(self, item):
        elems =  [self.format_date(date)for date in self.dates.__getitem__(item)]
        if len(elems) == 1:
            return elems[0]
        else:
            return elems



    def __contains__(self, date):
        return self.dates.__contains__(try_format(date, self.format))

    def exists(self):
        return True

    def format_date(self, date):
        return _dt.datetime.strftime(date, self.format)


    def select_date_range(self, start_date, end_date):
        """
        Selects and returns all the dates comprised between `start_date` and `end_date`

        Parameters
        ----------
        start_date : :obj:`str` or :obj:`_dt.datetime`
            The beginning of the date range
        end_date : :obj:`str` or :obj:`datetime.datetime`
            The end of the date range
        Returns
        -------
        :obj:`list` of :obj:`str`
            String of dates in the same format as the input dates
        """
        valid_dates = [self.format_date(date) for date in self.dates if
                       try_format(start_date, self.format) <= date <= try_format(end_date, self.format)]
        return valid_dates

    def select_with_distance(self, start_date, distance):
        stop_date = try_format(start_date, self.format) + _dt.timedelta(seconds=distance)
        valid_stop_date = self.select_n_dates(stop_date, 1)
        return  valid_stop_date


    def select_n_dates(self, start_date, n):
        """

        Selects and returns `n` dates starting from `start_date`

        Parameters
        ----------
        start_date : :obj:`str` or :obj:`_dt.datetime`
            The beginning of the date range
        n :  :obj:`int`
            The number of dates to select. If `n` is negative, returns dates starting from `start_date` backwards.
        Returns
        -------
        :obj:`list` of :obj:`str`
            String of dates in the same format as the input dates

        """
        comparison_op = _op.ge if n >= 0 else _op.le
        valid_dates = [self.format_date(date) for date in self.dates if
                       comparison_op(date, try_format(start_date, self.format))][:abs(n):]
        if len(valid_dates) == 1:
            valid_dates = valid_dates[0]
        return valid_dates

    def __repr__(self):
        return self.dates.__repr__()

    def all_combinations_with_separation(self, start_dt, stop_dt, bl_max=60 * 60):
        """
        This function computes all combinations
        of pairs of files between `start_dt` and `stop_dt` in the list that have a temporal
        separation less than `bl_max`

        Parameters
        ----------
        start_dt
        stop_dt
        bl_max

        Returns
        -------
        tuple of strings
        """
        # first find the nearest valid date to the start and to the stop
        correct_start = self.select_n_dates(start_dt, 1)
        correct_stop = self.select_n_dates(stop_dt, 1)
        # Now find all the dates in between
        dates = self.select_date_range(correct_start, correct_stop)
        # Compute unique combinations
        valid = [(master,slave) for master, slave in _iter.combinations(dates, 2) if
                 self.distance(master, slave) < bl_max]
        return valid


    def distance(self, start_date, stop_date):
        """
        Compute the distance in seconds beteween `start_date` and `stop_date`

        Parameters
        ----------
        start_date:  :obj:`str`, :obj:`_dt.datetime`
            The first date
        stop_date: :obj:`str`, :obj:`_dt.datetime`
            The second date
        Returns
        -------
            :ob:`float`
        """
        start = try_format(start_date, self.format)
        stop = try_format(stop_date, self.format)
        delta = (stop - start).total_seconds()
        return delta


class StackHelper:
    """
    Helper class to represent series of slcs,
    can use a Snakemake remoteprovider
    """

    @classmethod
    def fromfile(cls, fname, **kwargs):
        #parse dates
        ld = ListOfDates.fromfile(fname, **kwargs)
        c = StackHelper()
        c.all_dates = ld
        return c

    @classmethod
    def fromlist(cls, ld, **kwargs):
        ls = ListOfDates(ld, **kwargs)
        c = StackHelper()
        c.all_dates = ls
        return c

    def all_patterns_with_separation(self,wildcards, pattern, start_str='start_dt', stop_str='stop_dt',
                                   date_placeholder='datetime', bl_max=60 * 60):
        """
        This function computes all combinations
        of pairs of files between `wildcards.start_str` and `wildcards.stop_str` in the list that have a temporal
        separation less than `bl_max` and uses them to format `pattern`

        Parameters
        ----------
        wildcards
        pattern
        start_str
        stop_str
        bl_max

        Returns
        -------
        tuple of strings
        """
        valid_comb = self.all_dates.all_combinations_with_separation(wildcards[start_str], wildcards[stop_str], bl_max=bl_max)
        files = []
        wc = dict(wildcards)
        for m, s in valid_comb:
            wc[start_str] = m
            wc[stop_str] = s
            files.append(pattern.format(**wc))
        return files

    def nearest_valid(self, wildcards, pattern, start_str='start_dt'):
        """
        Return a string formatted according to the pattern `pattern` with all entries`{start_str}`
        substituted with the corresponding dates that are nearest to the date contained in `wildcards[start_str]`.

        Parameters
        ----------
        wildcards : :obj:`snakemake.wilcards`
            Dictionary of snakemake wildcards
        pattern : :obj:`str`
            The pattern to format, in the `format` format `{}{}`. It must contain
            `{start_str}` somewhere

        Returns
        -------
        `str`

        """
        start_dt_list = self.all_dates.select_n_dates(wildcards[start_str], 1)
        wc = dict(wildcards)
        wc[start_str] = start_dt_list[0]
        return pattern.format(**wc)

    def all_patterns_between_dates(self, wildcards, pattern, start_str='start_dt', stop_str='stop_dt',
                                   date_placeholder='datetime'):
        """
        Returns a list of strings formatted using the `pattern` pattern, with all entries `{date_placeholder}` subsituted
        with a valid date included between `wildcards[start_str]` and `wildcards[stop_str]`

        Parameters
        ----------
        wildcards : :obj:`snakemake.wilcards`
            Dictionary of snakemake wildcards, must contain `start_str` and `stop_str`
        pattern : :obj:`str`
            Pattern, must contain `date_placeholder` and the other entries in `wildcards`
        start_str : :obj:`str`
        stop_str : :obj:`str`

        Returns
        -------

        """
        dates = self.all_dates.select_date_range(wildcards[start_str], wildcards[stop_str])
        wc = dict(wildcards)
        strings = []
        for date in dates:
            wc[date_placeholder] = date
            strings.append(pattern.format(**wc))
        return strings

    def itab_entries_between_dates(self, wildcards, pattern, start_str='start_dt', stop_str='stop_dt', itab_path=None,
                                   **kwargs):
        """
        Returns all combinations of valid dates between the dates `start_str`
        and `end_str` computed using `Itab` with the itab parameters given in `**kwargs`. The dates are searched in the list of dates using the dates contained in `wildcards`
        . The combinations of dates
        are returned by substituing the elements `date_placeholders` in `pattern`.

        Parameters
        ----------
        wildcards : :obj:`snakemake.wilcards`
            Dictionary of snakemake wildcards
        pattern: :obj:`str`
            The pattern to format, in the `format` format `{}{}`. It must contain
            `date_placeholders[0]` and `date_placeholders[1]` somewhere

        Returns
        -------

        """
        # Get all dates between the specified dates
        dates = self.all_dates.select_date_range(wildcards[start_str], wildcards[stop_str])
        n_slc = len(dates)
        try:
            itab = Itab.fromfile(itab_path)
        except:
            try:
                itab = Itab(n_slc, **kwargs)
            except:
                raise AttributeError('Cannot create itab')
        wc = dict(wildcards)
        strings = []
        for master, slave, *rest in itab:
            wc[start_str] = dates[master - 1]
            wc[stop_str] = dates[slave - 1]
            strings.append(pattern.format(**wc))
        return strings

    def next_stack_dates(self, wildcards, pattern, start_str='start_dt', stop_str='stop_dt', index='i', **kwargs):
        itab = Itab(kwargs.pop('nstack'), **kwargs)
        # select next starting slc
        start_dt_list = self.all_dates.select_n_dates(wildcards[start_str], int(wildcards[index]))
        start_dt = start_dt_list[-1]
        # maxmium window length gives the last slc
        last_slc_index = _np.max(_np.array(itab.tab)[:, 0:2]) + 1
        # find the last slc
        stop_dt = self.all_dates.select_n_dates(start_dt, last_slc_index)[-1]
        # now select range
        valid_dates = self.all_dates.select_date_range(start_dt, stop_dt)
        wc = dict(wildcards)
        strings = []
        for master, slave, *rest in itab:
            wc[start_str] = valid_dates[master - 1]
            wc[stop_str] = valid_dates[slave - 1]
            strings.append(pattern.format(**wc))
        return strings


class Itab:
    """
    Class to represent itab files, list of acquisitions to compute
    interferograms
    """

    def __init__(self, n_slc, stride=1, step=1, n_ref=0, **kwargs):
        # number of slcs
        self.n_slc = n_slc
        self.tab = []
        stack_tab = []
        # The increment of the master slc
        self.stride = stride
        # The maximum number of steps between each master and slave
        self.max_distance = kwargs.get('max_distance', n_slc)
        # The size of each stack
        self.stack_size = kwargs.get('stack_size', n_slc)
        # The increment of the slave slc for every iteration
        self.step = step
        # the reference slc number
        self.n_ref = n_ref
        # itab line counter
        self.counter = 0
        self.it_counter = 0
        # Logic to select the list of reference slcs
        master = range(0, n_slc, stride) if stride > 0 else [n_ref] * n_slc
        #The slave increment of step with respect to the master
        slave = [master + step for master in master]
        tab = [[master, slave, counter, 1] for counter, (master,slave) in enumerate(zip(master, slave))]
        self.tab = tab
        # if stride == 0:  # if the master is not changing
        #     self.master = [self.n_ref,]
        #     # self.window = 0
        # else:
        #     self.master = iter(range(0, self.stack_size, stride))
        #
        # self.slave = range(self.step, self.step + self.max_distance, self.step)
        # for master, slave in _iter.product(self.master, self.slave):
        #     line = [master, slave + master]
        #     stack_tab.append(line)
        # list_of_slcs = list(range(n_slc))
        # line_counter = 0
        # for stack_counter, idx_stack in enumerate(range(0, self.n_slc // self.stack_size, 1)):
        #     for master, slave in stack_tab:
        #         master_idx = master + idx_stack * self.stack_size
        #         slave_idx = slave + idx_stack * self.stack_size
        #         if master_idx < self.n_slc and slave_idx < self.n_slc:
        #             line = [list_of_slcs[master_idx], list_of_slcs[slave_idx], line_counter, 1, stack_counter]
        #             self.tab.append(line)
        #             # # print(self.tab)
        #             line_counter += 1

    def __iter__(self):
        return iter(self.tab)

    def __str__(self):
        a = ""
        for line in self:
            line[0] += 1  # Add one
            line[1] += 1  # Add one because itab files are one-based indexed and python is zero based
            a = a + (" ".join(map(str, line)) + '\n')
        return a

    def __len__(self):
        return self.tab.__len__()

    @property
    def masters(self):
        return [t[0] for t in self.tab]

    @property
    def slaves(self):
        return [t[1] for t in self.tab]

    def __next__(self):
        try:
            el = self.tab[self.it_counter]
            self.it_counter += 1
            return el
        except IndexError:
            raise StopIteration

    def tofile(self, file):
        with open(file, 'w+') as of:
            of.write(print(self))
            # for line in self:
            #     line[0] += 1  # Add one
            #     line[1] += 1  # Add one because itab files are one-based indexed and python is zero based
            #     of.writelines(" ".join(map(str, line)) + " 1" + '\n')

    @staticmethod
    def pickle(file):
        with open(file, 'rb') as in_file:
            return pickle.load(in_file)

    def unpickle(self, file):
        with open(file, 'wb+') as of:
            pickle.dump(self, of, protocol=0)

    @staticmethod
    def fromfile(file):
        tab = _np.genfromtxt(file, dtype=int)
        step = tab[0, 0] - tab[1, 0]
        stride = tab[0, 1] - tab[1, 1]
        ref_slc = tab[0, 0] - 1
        n_slc = _np.max(tab[:, 0:2]) - 1
        n_stacks = _np.max(tab[:, -1]) + 1
        a = Itab(n_slc, step=step, stride=stride, n_ref=ref_slc, stack_size=n_stacks)
        tab[:, 0:2] = - 1  # subtract one because the file is saved with one based indices
        a.tab = tab
        return a

    def to_incidence_matrix(self):
        n_slc = self.n_slc
        A = _np.zeros((len(self.tab), n_slc))
        for idx_master, idx_slave, idx_itab, *rest in self:
            A[idx_itab - 1, idx_master - 1] = 1
            A[idx_itab - 1, idx_slave - 1] = -1
        return A


    def to_graph(self):
        G = _nx.Graph()
        for master, slave, *rest in self:
            G.add_edge(str(master), str(slave))
        return G

    def to_sbas_matrix(self):
        n_slc = self.n_slc
        A = _np.zeros((len(self.tab), n_slc))
        #For each interferogram
        for idx_master, idx_slave, idx_itab, *rest in self:
            #Find the ones between master and slave
            k = range(idx_slave-1, idx_master-1,-1)
            A[idx_itab-1, k] = (idx_slave + idx_master)
        return A
