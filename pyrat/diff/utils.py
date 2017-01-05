"""
Module with utilities to operate on list of SLCs;
especially useful in combination with snakemake to
select a certain range of dates, compute all interferograms etc

"""

import datetime as _dt
import operator as _op


def select_date_range(string_dates, date_start, date_end):
    dates = [_dt.datetime.strptime(x, dtfmt) for x in string_dates]
    dt_start = _dt.datetime.strptime(date_start, dtfmt)
    dt_end = _dt.datetime.strptime(date_end, dtfmt)
    valid_dates = [date.strftime(dtfmt) for date in dates if dt_start <= date <= dt_end]
    return valid_dates


def select_nth_previous(string_dates, date_end, n):
    dates = [_dt.datetime.strptime(x, dtfmt) for x in string_dates]
    dt_end = _dt.datetime.strptime(date_end, dtfmt)
    # find the closest element to the end date
    closest_index = min(dates, key=lambda x: abs(x - dt_end)) - 1
    start = closest_index - n
    valid_dates = dates[start]
    return valid_dates


def select_n_dates(string_dates, date_start, n_dates):
    dates = [_dt.datetime.strptime(x, dtfmt) for x in string_dates]
    dt_start = _dt.datetime.strptime(date_start, dtfmt)
    valid_dates = [date.strftime(dtfmt) for date in dates if dt_start <= date][:int(n_dates):]
    return valid_dates


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
        self.dates = [_dt.datetime.strptime(x, date_format) for x in dates]

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
        :obj:`list` of :obj:`datetime.datetime`

        """
        valid_dates = [date for date in self.dates if start_date <= date <= end_date]
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
        :obj:`list` of :obj:`datetime.datetime`

        """
        comparison_op = _op.ge if n > 0 else _op.le
        valid_dates = [date for date in self.dates if comparison_op(date, start_date)][:n:]
        return valid_dates


class StackHelper:
    """
    Helper class to represent
    """

    def __init__(self, list_of_slcs):
        with open(list_of_slcs, 'r') as f:  # the object contains a list of valid slcs
            reader = csv.reader(f)
            self.all_dates = list(reader)[0]
        self.itab = intfun.Itab(config['kalman']['nstack'], window=config['kalman']['nstack'],
                                step=config['ptarg']['step'], stride=config['ptarg']['stride'],
                                n_ref=config['ptarg']['ref']).tab

    def current_valid_date(self, wildcards, pattern):
        # return the nearest valid date given the "start_dt" wildcard
        start_dt_list = select_n_dates(self.all_dates, wildcards.start_dt, 1)
        wc = dict(wildcards)
        wc['start_dt'] = start_dt_list[0]
        return pattern.format(**wc)

    def next_stack_dates(self, wildcards):
        # select next starting slc
        start_dt_list = select_n_dates(self.all_dates, wildcards.start_dt, int(wildcards.i))
        start_dt = start_dt_list[-1]
        # maxmium window length gives the last slc
        last_slc_index = np.max(np.array(self.itab)[:, 0:2]) + 1
        # find the last slc
        stop_dt = select_n_dates(self.all_dates, start_dt, last_slc_index)[-1]
        # now select range
        valid_dates = select_date_range(self.all_dates, start_dt, stop_dt)
        #        print("Kalman Filter start date: {start_dt}".format(start_dt= wildcards.start_dt))
        #        print("Filter iteration index: {i}".format(i= wildcards.i))
        #        print("Last slc of stack: {stop_dt}".format(stop_dt=stop_dt))
        #        print("Slc in stack: {}".format(valid_dates))
        return valid_dates

    def next_stack_single(self, wildcards, pattern):
        valid_slcs = self.next_stack_dates(wildcards)
        return [pattern.format(datetime=s, chan=wildcards.chan) for s in valid_slcs]

    def next_stack_combinations(self, wildcards, pattern):
        valid_slcs = self.next_stack_dates(wildcards)
        # Compute ifgrams in itab
        ifgrams = []
        for master, slave, *rest in self.itab:
            ifgram = pattern.format(master=valid_slcs[master - 1], slave=valid_slcs[slave - 1], chan=wildcards.chan)
            ifgrams.append(ifgram)
        return ifgrams
