"""
Module with utilities to operate on list of SLCs;
especially useful in combination with snakemake to
select a certain range of dates, compute all interferograms etc

"""

import csv as _csv
import datetime as _dt
import operator as _op

import numpy as _np


def try_format(dt, fmt):
    try:
        dt.hour
    except AttributeError:
        return _dt.datetime.strptime(dt, fmt)
    else:
        return dt


class ListOfDates():
    """
    Helper class to represent a list of dates in a "Fuzzy" way.
    It permits 'soft' slicing: by slicing the object
    using a range of dates that may not be contained in the list, the closest date are chosen to slice the list of dates
    """

    def __init__(self, dates, date_format="%Y%m%d_%H%M%S"):
        """
        Initializes the ListOfDates object

        Parameters
        ----------
        dates : :obj:`iterable` of :obj:`str` or :obj:`datetime`
            List of dates either as a list of strings or of datetime objects
        date_format : :obj:`str`
            String to describe the parsing format, as used by datetime.datetime.strptime

        """
        self.dates = sorted([_dt.datetime.strptime(x, date_format) for x in dates])
        self.format = date_format

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
        valid_dates = [date.strftime(self.format) for date in self.dates if
                       try_format(start_date, self.format) <= date <= try_format(end_date, self.format)]
        return valid_dates

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
        valid_dates = [date.strftime(self.format) for date in self.dates if
                       comparison_op(date, try_format(start_date, self.format))][:abs(n):]
        return valid_dates


class StackHelper:
    """
    Helper class to represent series of slcs
    """

    def __init__(self, list_of_slcs, nstack=1, window=None, n_ref=0, stride=1, step=1, **kwargs):
        # Load list of slcs
        try:
            with open(list_of_slcs, 'r') as f:  # the object contains a list of valid slcs
                reader = _csv.reader(f)
                self.all_dates = ListOfDates(list(reader)[0], **kwargs)
        except (FileNotFoundError, TypeError):
            try:
                self.all_dates = ListOfDates(list_of_slcs, **kwargs)
            except:
                raise TypeError("{l} is not a file path or a list of strings".format(list_of_slcs))

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


    def all_patterns_between_dates(self, wildcards, pattern, start_str='start_dt', stop_str='stop_dt', date_placeholder='datetime'):
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

    def itab_entries_between_dates(self, wildcards, pattern, start_str='start_dt', stop_str='stop_dt', **kwargs):
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
        #Get all dates between the specified dates
        dates = self.all_dates.select_date_range(wildcards[start_str], wildcards[stop_str])
        n_slc = len(dates)
        itab = Itab(n_slc, **kwargs)
        wc = dict(wildcards)
        strings = []
        for master, slave, *rest in itab:
            wc[start_str] = dates[master - 1]
            wc[stop_str] = dates[slave - 1]
            strings.append(pattern.format(**wc))
        print(strings)
        return strings

    def nearest_n(self, wilcards, pattern, index='i', start_str='start_dt'):
        """
        Return a list of strings formatted according to the pattern `pattern` with all entries`{start_str}`
        substituted with the `index` dates before `wildcards[`start_dt`]`.

        Parameters
        ----------
        wildcards : :obj:`snakemake.wilcards`
            Dictionary of snakemake wildcards
        pattern : :obj:`str`
            The pattern to format, in the `format` format `{}{}`. It must contain
            `{start_str}` somewhere

        Returns
        -------
        :obj:`list` of :obj:`str`

        """
        start_dt_list = self.all_dates.select_n_dates(wildcards[start_str], 1)
        wc = dict(wildcards)
        wc[start_str] = start_dt_list[0]
        return pattern.format(**wc)


        #
        # def next_stack_dates(self, wildcards):
        #     # select next starting slc
        #     start_dt_list = select_n_dates(self.all_dates, wildcards.start_dt, int(wildcards.i))
        #     start_dt = start_dt_list[-1]
        #     # maxmium window length gives the last slc
        #     last_slc_index = np.max(np.array(self.itab)[:, 0:2]) + 1
        #     # find the last slc
        #     stop_dt = select_n_dates(self.all_dates, start_dt, last_slc_index)[-1]
        #     # now select range
        #     valid_dates = select_date_range(self.all_dates, start_dt, stop_dt)
        #     #        print("Kalman Filter start date: {start_dt}".format(start_dt= wildcards.start_dt))
        #     #        print("Filter iteration index: {i}".format(i= wildcards.i))
        #     #        print("Last slc of stack: {stop_dt}".format(stop_dt=stop_dt))
        #     #        print("Slc in stack: {}".format(valid_dates))
        #     return valid_dates
        #
        # def next_stack_single(self, wildcards, pattern):
        #     valid_slcs = self.next_stack_dates(wildcards)
        #     return [pattern.format(datetime=s, chan=wildcards.chan) for s in valid_slcs]
        #
        # def next_stack_combinations(self, wildcards, pattern):
        #     valid_slcs = self.next_stack_dates(wildcards)
        #     # Compute ifgrams in itab
        #     ifgrams = []
        #     for master, slave, *rest in self.itab:
        #         ifgram = pattern.format(master=valid_slcs[master - 1], slave=valid_slcs[slave - 1], chan=wildcards.chan)
        #         ifgrams.append(ifgram)
        #     return ifgrams


class Itab:
    """
    Class to represent itab files, list of acquisitions to compute
    interferograms
    """

    def __init__(self, n_slc, stride=1, window=None, step=1, n_ref=0, **kwargs):
        self.tab = []
        # The increment of the master slc
        self.stride = stride
        # The maximum number of steps between each master and slave
        self.window = 1 or window
        # The increment of the slave slc for every iteration
        self.step = step
        # the reference slc number
        self.n_ref = 0
        # itab line counter
        self.counter = 1
        self.it_counter = 0
        # Logic to select the list of reference slcs
        if stride == 0:  # if the master is not changing
            self.reference = (x for x in n_ref)
            self.window = 0
        else:
            self.reference = iter(range(n_ref, n_slc, stride))
        # Counter of slaves
        for master in self.reference:
            for slave in range(master + self.step, master + self.step + self.window, self.step):
                self.counter += 1
                line = [master, slave, self.counter]
                self.tab.append(line)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            el = self.tab[self.it_counter]
            self.it_counter += 1
            return el
        except IndexError:
            raise StopIteration

    def tofile(self, file):
        with open(file, 'w+') as of:
            for line in self:
                of.writelines(" ".join(map(str, line)) + " 1" + '\n')

    @staticmethod
    def fromfile(file):
        tab = _np.genfromtxt(file, dtype=int)
        step = tab[0, 0] - tab[1, 0]
        stride = tab[0, 1] - tab[1, 1]
        ref_slc = tab[0, 0]
        n_slc = _np.max(tab[:, 0:2])
        a = Itab(n_slc, step=step, stride=stride, n_ref=ref_slc)
        a.tab = tab
        return a

    def to_incidence_matrix(self):
        n_slc = self.n_slc
        A = _np.zeros((len(self.tab), n_slc + 1))
        for idx_master, idx_slave, idx_itab, *rest in self.tab:
            A[idx_itab - 1, idx_master] = 1
            A[idx_itab - 1, idx_slave] = -1
        return A
