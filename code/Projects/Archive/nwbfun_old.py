"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import datetime
import gc
import os
import time
from pathlib import Path
import pynapple as nap

import numpy as np
import scipy
from natsort import natsorted

from suite2p import run_s2p
from suite2p.detection.stats import roi_stats
from suite2p.detection import utils
from suite2p import run_s2p, default_ops

try:
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries, validate
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import (
        Fluorescence,
        ImageSegmentation,
        OpticalChannel,
        RoiResponseSeries,
        TwoPhotonSeries,
    )

    from pynwb.behavior import (
        BehavioralEpochs,
        BehavioralEvents,
        BehavioralTimeSeries,
    )    
    from pynwb.epoch import TimeIntervals
    from pynwb.misc import IntervalSeries

    NWB = True
except ModuleNotFoundError:
    NWB = False

# a class designed to interact with suite2p formats
class suite2p_nwb():

    def __init__(self):
        pass 

    def read_nwb(self, fpath):
        """read NWB file for use in the GUI"""
        with NWBHDF5IO(fpath, "r") as fio:
            nwbfile = fio.read()

            # ROIs
            try:
                rois = nwbfile.processing["ophys"]["ImageSegmentation"][
                    "PlaneSegmentation"]["pixel_mask"]
                multiplane = False
            except Exception:
                rois = nwbfile.processing["ophys"]["ImageSegmentation"][
                    "PlaneSegmentation"]["voxel_mask"]
                multiplane = True
            stat = []
            for n in range(len(rois)):
                if isinstance(rois[0], np.ndarray):
                    stat.append({
                        "ypix":
                            np.array(
                                [rois[n][i][0].astype("int") for i in range(len(rois[n]))]),
                        "xpix":
                            np.array(
                                [rois[n][i][1].astype("int") for i in range(len(rois[n]))]),
                        "lam":
                            np.array([rois[n][i][-1] for i in range(len(rois[n]))]),
                    })
                else:
                    stat.append({
                        "ypix": rois[n]["x"].astype("int"),
                        "xpix": rois[n]["y"].astype("int"),
                        "lam": rois[n]["weight"],
                    })
                if multiplane:
                    stat[-1]["iplane"] = int(rois[n][0][-2])
            ops = default_ops()

            if multiplane:
                nplanes = (np.max(np.array([stat[n]["iplane"] for n in range(len(stat))])) +
                        1)
            else:
                nplanes = 1
            stat = np.array(stat)

            # ops with backgrounds
            ops1 = []
            for iplane in range(nplanes):
                ops = default_ops()
                bg_strs = ["meanImg", "Vcorr", "max_proj", "meanImg_chan2"]
                ops["nchannels"] = 1
                for bstr in bg_strs:
                    if (bstr in nwbfile.processing["ophys"]["Backgrounds_%d" %
                                                            iplane].images):
                        ops[bstr] = np.array(nwbfile.processing["ophys"]["Backgrounds_%d" %
                                                                        iplane][bstr].data)
                        if bstr == "meanImg_chan2":
                            ops["nchannels"] = 2
                ops["Ly"], ops["Lx"] = ops[bg_strs[0]].shape
                ops["yrange"] = [0, ops["Ly"]]
                ops["xrange"] = [0, ops["Lx"]]
                ops["tau"] = 1.0
                ops["fs"] = nwbfile.acquisition["TwoPhotonSeries"].rate
                ops1.append(ops.copy())

            stat = roi_stats(stat, ops["Ly"], ops["Lx"], ops["aspect"], ops["diameter"])

            # fluorescence
            ophys = nwbfile.processing["ophys"]

            def get_fluo(name: str) -> np.ndarray:
                """Extract Fluorescence data."""
                roi_response_series = ophys[name].roi_response_series
                if name in roi_response_series.keys():
                    fluo = ophys[name][name].data[:]
                elif "plane0" in roi_response_series.keys():
                    for key, value in roi_response_series.items():
                        if key == "plane0":
                            fluo = value.data[:]
                        else:
                            fluo = np.concatenate((fluo, value.data[:]), axis=1)
                    fluo = np.transpose(fluo)
                else:
                    raise AttributeError(f"Can't find {name} container in {fpath}")
                return fluo

            F = get_fluo("Fluorescence")
            Fneu = get_fluo("Neuropil")
            spks = get_fluo("Deconvolved")
            dF = F - ops["neucoeff"] * Fneu
            for n in range(len(stat)):
                stat[n]["skew"] = scipy.stats.skew(dF[n])

            # cell probabilities
            iscell = [
                ophys["ImageSegmentation"]["PlaneSegmentation"]["iscell"][n]
                for n in range(len(stat))
            ]
            iscell = np.array(iscell)
            probcell = iscell[:, 1]
            iscell = iscell[:, 0].astype("bool")
            redcell = np.zeros_like(iscell)
            probredcell = np.zeros_like(probcell)

            if multiplane:
                ops = ops1[0].copy()
                Lx = ops["Lx"]
                Ly = ops["Ly"]
                nX = np.ceil(np.sqrt(ops["Ly"] * ops["Lx"] * len(ops1)) / ops["Lx"])
                nX = int(nX)
                for j in range(len(ops1)):
                    ops1[j]["dx"] = (j % nX) * Lx
                    ops1[j]["dy"] = int(j / nX) * Ly

                LY = int(np.amax(np.array([ops["Ly"] + ops["dy"] for ops in ops1])))
                LX = int(np.amax(np.array([ops["Lx"] + ops["dx"] for ops in ops1])))
                meanImg = np.zeros((LY, LX))
                max_proj = np.zeros((LY, LX))
                if ops["nchannels"] > 1:
                    meanImg_chan2 = np.zeros((LY, LX))

                Vcorr = np.zeros((LY, LX))
                for k, ops in enumerate(ops1):
                    xrange = np.arange(ops["dx"], ops["dx"] + ops["Lx"])
                    yrange = np.arange(ops["dy"], ops["dy"] + ops["Ly"])
                    meanImg[np.ix_(yrange, xrange)] = ops["meanImg"]
                    Vcorr[np.ix_(yrange, xrange)] = ops["Vcorr"]
                    max_proj[np.ix_(yrange, xrange)] = ops["max_proj"]
                    if ops["nchannels"] > 1:
                        if "meanImg_chan2" in ops:
                            meanImg_chan2[np.ix_(yrange, xrange)] = ops["meanImg_chan2"]
                    for j in np.nonzero(
                            np.array([stat[n]["iplane"] == k for n in range(len(stat))
                                    ]))[0]:
                        stat[j]["xpix"] += ops["dx"]
                        stat[j]["ypix"] += ops["dy"]
                        stat[j]["med"][0] += ops["dy"]
                        stat[j]["med"][1] += ops["dx"]
                ops["Vcorr"] = Vcorr
                ops["max_proj"] = max_proj
                ops["meanImg"] = meanImg
                if "meanImg_chan2" in ops:
                    ops["meanImg_chan2"] = meanImg_chan2
                ops["Ly"], ops["Lx"] = LY, LX
                ops["yrange"] = [0, LY]
                ops["xrange"] = [0, LX]
        return stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell

    def save_nwb(self, datafolder: str, nwbsavename: str = "ophys.nwb", savefolder = None, brain_region: str = 'PFC', lab='Spellman', institution='UCONN Health', session_description='2P recording', raw_sessions_file = None):
        """convert folder with plane folders to NWB format
        
            Args:
                >>> datafolder: string directory to the suite2p folder in a session

            Optional Args:
                >>> nwbsavename: name of NWB file to save. Default is "ophys.nwb"
                >>> savefolder: location to save the NWB file
                >>> brain_region: which structure you are recording from

            Future versions can take in a .raw sessions file
        """

        plane_folders = natsorted([
            Path(f.path)
            for f in os.scandir(datafolder)
            if f.is_dir() and f.name[:5] == "plane"
        ])
        ops1 = [
            np.load(f.joinpath("ops.npy"), allow_pickle=True).item() for f in plane_folders
        ]
        nchannels = min([ops["nchannels"] for ops in ops1])

        if NWB and not ops1[0]["mesoscan"]:
            if len(ops1) > 1:
                multiplane = True
            else:
                multiplane = False

            ops = ops1[0]
            if "date_proc" in ops:
                session_start_time = ops["date_proc"]
                if not session_start_time.tzinfo:
                    session_start_time = session_start_time.astimezone()
            else:
                session_start_time = datetime.datetime.now().astimezone()

            # path to data = save folder
            ops["data_path"] = datafolder
            ops["save_path"] = datafolder

            # INITIALIZE NWB FILE
            nwbfile = NWBFile(
                session_description=session_description,
                identifier=str(ops["data_path"][0]),
                session_start_time=session_start_time,
                lab=lab,
                institution=institution
            )
            print(nwbfile)

            device = nwbfile.create_device(
                name="Microscope",
                description="My two-photon microscope",
                manufacturer="ThorLabs",
            )
            optical_channel = OpticalChannel(
                name="OpticalChannel",
                description="an optical channel",
                emission_lambda=500.0,
            )

            imaging_plane = nwbfile.create_imaging_plane(
                name="ImagingPlane",
                optical_channel=optical_channel,
                imaging_rate=ops["fs"],
                description="standard",
                device=device,
                excitation_lambda=600.0,
                indicator="GCaMP",
                location="PFC",
                grid_spacing=([2.0, 2.0, 30.0] if multiplane else [2.0, 2.0]), # check on this
                grid_spacing_unit="microns",
            )

            # get timestamps
            numtimes = ops['frames_per_file'][0]
            relative_start = 0
            relative_end = numtimes/ops['fs'] # num samples/frame rate = number of sec
            timestamps = np.linspace(start=relative_start, stop=relative_end,num=numtimes)

            # link to external data
            external_data = [""]
            #external_data = os.path.join(save_folder, os.path.split(external_data)[-1])
            image_series = TwoPhotonSeries(
                name="TwoPhotonSeries",
                dimension=[ops["Ly"], ops["Lx"]],
                external_file=external_data,
                imaging_plane=imaging_plane,
                starting_frame=[0 for i in range(len(external_data))],
                format="external",
                starting_time=0.0,
                rate=ops["fs"] * ops["nplanes"],
            )
            nwbfile.add_acquisition(image_series)

            # processing
            img_seg = ImageSegmentation()
            ps = img_seg.create_plane_segmentation(
                name="PlaneSegmentation",
                description="suite2p output",
                imaging_plane=imaging_plane,
                reference_images=image_series,
            )
            ophys_module = nwbfile.create_processing_module(
                name="ophys", description="optical physiology processed data")
            ophys_module.add(img_seg)

            file_strs = ["F.npy", "Fneu.npy", "spks.npy"]
            file_strs_chan2 = ["F_chan2.npy", "Fneu_chan2.npy"]
            traces, traces_chan2 = [], []
            ncells = np.zeros(len(ops1), dtype=np.int_)
            Nfr = np.array([ops["nframes"] for ops in ops1]).max()
            for iplane, ops in enumerate(ops1):
                if iplane == 0:
                    iscell = np.load(os.path.join(ops["save_path"],'plane0', "iscell.npy"))
                    for fstr in file_strs:
                        traces.append(np.load(os.path.join(ops["save_path"],'plane0', fstr)))
                    if nchannels > 1:
                        for fstr in file_strs_chan2:
                            traces_chan2.append(
                                np.load(plane_folders[iplane].joinpath(fstr)))
                    PlaneCellsIdx = iplane * np.ones(len(iscell))
                else:
                    iscell = np.append(
                        iscell,
                        np.load(os.path.join(ops["save_path"],'plane0',"iscell.npy")),
                        axis=0,
                    )
                    for i, fstr in enumerate(file_strs):
                        trace = np.load(os.path.join(ops["save_path"],'plane0',fstr))
                        if trace.shape[1] < Nfr:
                            fcat = np.zeros((trace.shape[0], Nfr - trace.shape[1]),
                                            "float32")
                            trace = np.concatenate((trace, fcat), axis=1)
                        traces[i] = np.append(traces[i], trace, axis=0)
                    if nchannels > 1:
                        for i, fstr in enumerate(file_strs_chan2):
                            traces_chan2[i] = np.append(
                                traces_chan2[i],
                                np.load(plane_folders[iplane].joinpath(fstr)),
                                axis=0,
                            )
                    PlaneCellsIdx = np.append(
                        PlaneCellsIdx, iplane * np.ones(len(iscell) - len(PlaneCellsIdx)))

                stat = np.load(os.path.join(ops["save_path"],'plane0',"stat.npy"),
                            allow_pickle=True)
                ncells[iplane] = len(stat)
                for n in range(ncells[iplane]):
                    if multiplane:
                        pixel_mask = np.array([
                            stat[n]["ypix"],
                            stat[n]["xpix"],
                            iplane * np.ones(stat[n]["npix"]),
                            stat[n]["lam"],
                        ])
                        ps.add_roi(voxel_mask=pixel_mask.T)
                    else:
                        pixel_mask = np.array(
                            [stat[n]["ypix"], stat[n]["xpix"], stat[n]["lam"]])
                        ps.add_roi(pixel_mask=pixel_mask.T)

            ps.add_column("iscell", "two columns - iscell & probcell", iscell)

            rt_region = []
            for iplane, ops in enumerate(ops1):
                if iplane == 0:
                    rt_region.append(
                        ps.create_roi_table_region(
                            region=list(np.arange(0, ncells[iplane]),),
                            description=f"ROIs for plane{int(iplane)}",
                        ))
                else:
                    rt_region.append(
                        ps.create_roi_table_region(
                            region=list(
                                np.arange(
                                    np.sum(ncells[:iplane]),
                                    ncells[iplane] + np.sum(ncells[:iplane]),
                                )),
                            description=f"ROIs for plane{int(iplane)}",
                        ))

            # FLUORESCENCE (all are required)
            name_strs = ["Fluorescence", "Neuropil", "Deconvolved"]
            name_strs_chan2 = ["Fluorescence_chan2", "Neuropil_chan2"]

            for i, (fstr, nstr) in enumerate(zip(file_strs, name_strs)):
                for iplane, ops in enumerate(ops1):
                    roi_resp_series = RoiResponseSeries(
                        name=f"plane{int(iplane)}",
                        data=np.transpose(traces[i][PlaneCellsIdx == iplane]),
                        rois=rt_region[iplane],
                        unit="lumens",
                        rate=ops["fs"],
                    )
                    if iplane == 0:
                        fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
                    else:
                        fl.add_roi_response_series(roi_response_series=roi_resp_series)
                ophys_module.add(fl)

            if nchannels > 1:
                for i, (fstr, nstr) in enumerate(zip(file_strs_chan2, name_strs_chan2)):
                    for iplane, ops in enumerate(ops1):
                        roi_resp_series = RoiResponseSeries(
                            name=f"plane{int(iplane)}",
                            data=np.transpose(traces_chan2[i][PlaneCellsIdx == iplane]),
                            rois=rt_region[iplane],
                            unit="lumens",
                            rate=ops["fs"],
                            timestamps=timestamps # might need to cut
                        )

                        if iplane == 0:
                            fl = Fluorescence(roi_response_series=roi_resp_series,
                                            name=nstr)
                        else:
                            fl.add_roi_response_series(roi_response_series=roi_resp_series)

                    ophys_module.add(fl)

            # BACKGROUNDS
            # (meanImg, Vcorr and max_proj are REQUIRED)
            bg_strs = ["meanImg", "Vcorr", "max_proj", "meanImg_chan2"]
            for iplane, ops in enumerate(ops1):
                images = Images("Backgrounds_%d" % iplane)
                for bstr in bg_strs:
                    if bstr in ops:
                        if bstr == "Vcorr" or bstr == "max_proj":
                            img = np.zeros((ops["Ly"], ops["Lx"]), np.float32)
                            img[
                                ops["yrange"][0]:ops["yrange"][-1],
                                ops["xrange"][0]:ops["xrange"][-1],
                            ] = ops[bstr]
                        else:
                            img = ops[bstr]
                        images.add_image(GrayscaleImage(name=bstr, data=img))
                ophys_module.add(images)

            #imaging_module = nwbfile.create_processing_module(
                #name="SummaryImages", description="Backgrounds_0 summary images")
            #imaging_module.add(images)

            if savefolder is None:
                savepath = os.path.join(datafolder, nwbsavename)
                with NWBHDF5IO(savepath, "w") as fio:
                    fio.write(nwbfile)
                    fio.close()
            else:
                savepath = os.path.join(savefolder, nwbsavename)
                with NWBHDF5IO(savepath, "w") as fio:
                    fio.write(nwbfile) 
                    fio.close()           
        else:
            print("pip install pynwb OR don't use mesoscope recording")

        return savepath
    
    def neuroconv_nwb():
        pass

# a class designed to work with our behavior data
class behavior_nwb():
    """
    This class structures your behavioral file.

    John Stout
    """

    def __init__(self, matpath: str):
        """
            Args:
                >>> matpath: path to your .mat behavioral file
            
            Returns:
                >>> behdata: behavioral data loaded
                >>> varnames: variable names
        """
        self.behdata  = scipy.io.loadmat(matpath)
        self.varnames = [i for i in self.behdata if '__' not in i]

    def autoassigndict(self):
        '''
        Saves information from your dictionary based on the name of the variable
        '''

        # Convert the numpy array to a Python dictionary
        matstruct = dict()
        for keys in self.varnames:
            if self.behdata[keys].dtype.names is not None:
                matstruct[keys] = dict()
                for field in self.behdata[keys].dtype.names:
                    for array in self.behdata[keys][field]:
                        for value in array:
                            matstruct[keys][field] = value
            else:
                if type(self.behdata[keys]) is np.ndarray or type(self.behdata[keys]) is np.array:
                    matstruct[keys] = self.behdata[keys]

        self.behdict = matstruct

        # now align trials
        trial_start_times = self.behdict['trialStartTimes'][0]
        trial_end_times   = self.behdict['trialEndTimes'][0]

        # if there are more end times than there are start times
        if len(trial_end_times) > len(trial_start_times):

            # more common than not, you will find an extra end time
            trial_end_times = trial_end_times[0:-1]
            print("End trial shaved off")
        
        assert len(trial_end_times) == len(trial_start_times), "There are an uneven number of start and stop times"

        # if there are just as many end as there are start times
        if len(trial_end_times) == len(trial_start_times):

            # concatenate
            trials_cat = np.concatenate(([trial_start_times], [trial_end_times]),axis=0).T

            # check rows/columns
            if trials_cat.shape[0] < trials_cat.shape[1]:
                trials_cat = trials_cat.T
            
            # calculate trial durations
            trial_durations = trials_cat[:,1]-trials_cat[:,0]

            # search for negative values
            trials_misaligned = (trial_durations < 0).any()
            assert trials_misaligned==False, "Start and stop times are not aligned. Negative values detected (stop-start) indicating stops that occured after a start."

        # trial duration info
        self.behdict['trialSampleLength']  = [trial_durations]
        self.behdict['trialStartTimes']    = [trial_start_times]
        self.behdict['trialEndTimes']      = [trial_end_times]
        self.behdict['trials_cat']         = [trials_cat]

        return matstruct

    def readwrite(self, existing_nwb_path: str):
        """
        This code takes an existing .nwb file, and writes your behavioral data to it.
        
        John Stout
        """

        # read existing file
        with NWBHDF5IO(existing_nwb_path, "r+") as io:
            #io = NWBHDF5IO(existing_nwb_path, mode="a") # read in editor mode
            nwbfile = io.read()

            # get data into a dictionary to loop over
            matstruct = self.autoassigndict()

            # make sure your fluorescence data is aligned to your timestamps
            F = nwbfile.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['plane0'].data[:]
            srate = nwbfile.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['plane0'].rate

            # make sure that your F and timestamps align
            assert F.shape[0] == matstruct['frameTimes'].shape[1], "Fluorescence data is not aligned to your behavioral timestamps"

            # create behavioral module
            #behavior_module = nwbfile.create_processing_module(
            #    name="behavior", description="Processed behavioral data"
            #)

            # convert your frametimes, treating your first frameTime as recording start
            #start_time = self.behdict['frameTimes'][0][0]; 
            #end_time   = self.behdict['frameTimes'][0][-1]; 

            numtimes = self.behdict['frameTimes'][0].shape[0]
            relative_start = 0
            relative_end = numtimes/srate # num samples/frame rate/secinmin
            timestamps = np.linspace(start=relative_start, stop=relative_end, num=numtimes)

            # timestamps for frames
            time_series = TimeSeries(
                name="frameTimes",
                data=matstruct['frameTimes'][0],
                #rate=srate,
                timestamps=timestamps,
                description="frameTimes are actual timestamps. Timestamps added are relative from 0 to time end",
                unit="s", # I don't know the unit
            )
            nwbfile.add_acquisition(time_series)

            # data as acquisition????

            # -- create interval series object -- #
            '''
            # automate later
            behavioral_epochs = BehavioralEpochs(name="BehavioralEpochs")

            # TRIAL START TIMES
            trialStart_idx_series = IntervalSeries(
                name="trial_start_index",
                data=self.behdict['trialStartTimes'][0],
                timestamps=timestamps,
                description="Index of trial start times"
            )
            behavioral_epochs.add_interval_series(trialStart_idx_series)

            # TRIAL END TIMES
            trialEnd_idx_series = IntervalSeries(
                name="trial_end_index",
                data=self.behdict['trialEndTimes'][0],
                timestamps=timestamps,
                description="Index of trial end times"
            )
            behavioral_epochs.add_interval_series(trialEnd_idx_series)

            # REWARD TIMES INDEX
            reward_idx_series = IntervalSeries(
                name="reward_time_index",
                data=self.behdict['rewardTimes'][0],
                timestamps=timestamps,
                description="Index of reward times"
            )
            behavioral_epochs.add_interval_series(reward_idx_series)
            '''
            # identify correct trial and incorrect trial (motivational) reward times
            #trial_times = np.concatenate(([self.behdict['trialStartTimes'][0]], [self.behdict['trialEndTimes'][0][0:-1]]),axis=0).T
            trial_times = self.behdict['trials_cat'][0]
            trialRewardCorrectTimes = []; trialRewardIncorrectTimes = []; trialRewardIdx = []; 
            trialRewardTime = np.empty((trial_times.shape[0],)); trialRewardTime[:]=np.nan
            trialRewardTimeIdx = np.empty((trial_times.shape[0],)); trialRewardTimeIdx[:]=np.nan
            for triali, trialtimes in enumerate(trial_times):
                trial_found = 0
                for rewti in self.behdict['rewardTimes'][0]:
                    if rewti > trialtimes[0] and rewti <= trialtimes[1] and self.behdict['trialCorrect'][0][triali]==1: # if reward time falls in between trial
                        trialRewardCorrectTimes.append(rewti)
                        trial_found = 1
                        trialRewardTimeIdx[triali]=rewti
                        trialRewardTime[triali]=timestamps[rewti]
                    elif rewti > trialtimes[0] and rewti <= trialtimes[1] and self.behdict['trialCorrect'][0][triali]==0:
                        trialRewardIncorrectTimes.append(rewti) # times for reward
                        trial_found = 1 # marker for whether reward happened or not
                        trialRewardTimeIdx[triali]=rewti
                        trialRewardTime[triali]=timestamps[rewti]
                trialRewardIdx.append(trial_found)
            trialRewardCorrectTimes   = np.array(trialRewardCorrectTimes)
            trialRewardIncorrectTimes = np.array(trialRewardIncorrectTimes)
            trialRewardIdx            = np.array(trialRewardIdx)
            '''
            # REWARD TIMES INDEX
            reward_times_all_idx_series = IntervalSeries(
                name="trial_reward_times",
                data=trialRewardTime,
                timestamps=timestamps,
                description="Index of reward times organized by trial"
            )
            behavioral_epochs.add_interval_series(reward_times_all_idx_series)

            # REWARD TIMES INDEX
            reward_correct_idx_series = IntervalSeries(
                name="trial_correct_reward_times",
                data=trialRewardCorrectTimes,
                timestamps=timestamps,
                description="Index of reward times during correct trials"
            )
            behavioral_epochs.add_interval_series(reward_correct_idx_series)

            # REWARD TIMES INDEX
            reward_incorrect_idx_series = IntervalSeries(
                name="trial_incorrect_reward_times",
                data=trialRewardIncorrectTimes,
                timestamps=timestamps,
                description="Index of reward times during incorrect trials"
            )
            behavioral_epochs.add_interval_series(reward_incorrect_idx_series)

            # STIM ON
            stimOn_idx_series = IntervalSeries(
                name="stim_on_times",
                data=self.behdict['stimOnTimes'][0],
                timestamps=timestamps,
                description="Index of stimulus onset"
            )
            behavioral_epochs.add_interval_series(stimOn_idx_series)        

            # STIM OFF
            stimOff_idx_series = IntervalSeries(
                name="stim_off_times",
                data=self.behdict['stimOffTimes'][0],
                timestamps=timestamps,
                description="Index of stimulus offset"
            )
            behavioral_epochs.add_interval_series(stimOff_idx_series)  

            # Lick times L - this tells us the number of licks occuring between timestamps
            lickTimesL_idx_series = IntervalSeries(
                name="lick_times_left_index",
                data=self.behdict['lickTimesL'][0],
                timestamps=timestamps,
                description="Index of lick times. Relate this to trial_start_times and trial_end_times. If trial start index is 20 and end is 28, but you have 5 repeats of 26, then you know that 5 licks happened on the first trial"
            )
            behavioral_epochs.add_interval_series(lickTimesL_idx_series)       

            # Lick times R - this tells us the number of licks occuring between timestamps
            lickTimesR_idx_series = IntervalSeries(
                name="lick_times_right_index",
                data=self.behdict['lickTimesR'][0],
                timestamps=timestamps,
                description="Index of lick times. Relate this to trial_start_times and trial_end_times. If trial start index is 20 and end is 28, but you have 5 repeats of 26, then you know that 5 licks happened on the first trial"
            )
            behavioral_epochs.add_interval_series(lickTimesR_idx_series)   

            # Lick times all - this tells us the number of licks occuring between timestamps
            lickTimesAll_idx_series = IntervalSeries(
                name="lick_times_all_index",
                data=self.behdict['lickTimesAll'][0],
                timestamps=timestamps,
                description="Index of lick times. Relate this to trial_start_times and trial_end_times. If trial start index is 20 and end is 28, but you have 5 repeats of 26, then you know that 5 licks happened on the first trial"
            )
            behavioral_epochs.add_interval_series(lickTimesAll_idx_series)                         

            # now add everything to the NWBFile
            behavior_module.add(behavioral_epochs)
            '''
            
            # -- NOW WE ADD TRIAL STRUCTURE FOR COUNT DATA -- # 
            nwbfile.add_trial_column(name="trialStartIdx",         description="Index of trial start times - use with frameTimes or imaging data")    
            nwbfile.add_trial_column(name="trialEndIdx",           description="Index of trial start times - use with frameTimes or imaging data")

            nwbfile.add_trial_column(name="LickCountsL",           description="# of left licks")
            nwbfile.add_trial_column(name="LickCountsR",           description="# of right licks")

            nwbfile.add_trial_column(name="irrelevantLR",          description="Boolean: which side (Left vs Right) the irrelevant stimulus was present")  
            nwbfile.add_trial_column(name="relevantLR",            description="Boolean: which side (Left vs Right) the relevant stimulus was present") 
            nwbfile.add_trial_column(name="setShiftingID",         description="Boolean: Attentional set (Odor Vs Whisker)") 
            nwbfile.add_trial_column(name="choiceOutcome",         description="Boolean: Correct (1) vs incorrect (0)") 
            nwbfile.add_trial_column(name="trialRewarded",         description="Boolean: Rewarded (1) vs not rewarded (0)") 

            nwbfile.add_trial_column(name="trialDuration",         description="Duration of trials (seconds)")  
            nwbfile.add_trial_column(name="stimDuration",          description="Duration of stimulation (in seconds)") 

            nwbfile.add_trial_column(name="trialRewardTimesIdx",   description="Index of reward times")                         
            nwbfile.add_trial_column(name="trialRewardTimes",      description="Converted reward times (seconds)")

            nwbfile.add_trial_column(name="stimulusOnsetIdx",      description="Index of stimulus onset")                         
            nwbfile.add_trial_column(name="stimulusOnsetTime",     description="Converted stimulus onset (seconds)")

            nwbfile.add_trial_column(name="stimulusOffsetIdx",     description="Index of stimulus offset")                         
            nwbfile.add_trial_column(name="stimulusOffsetTime",    description="Converted stimulus offset (seconds)")

            # loop over trials - FACT CHECK ME
            start_times   = timestamps[self.behdict['trialStartTimes'][0]]
            end_times     = timestamps[self.behdict['trialEndTimes'][0]]
            stim_on       = timestamps[self.behdict['stimOnTimes'][0]]
            stim_off      = timestamps[self.behdict['stimOffTimes'][0]]
            stim_duration = stim_off-stim_on
            #iti = stim_off[0:-1]-stim_on[1::]

            for triali in range(len(self.behdict['trialNum'][0])):

                # lick left counts
                #assert stim_off[triali] < end_times[triali], "Stimulus offset did not exceed trial end time"
                nwbfile.add_trial(start_time=start_times[triali], stop_time=end_times[triali], 
                                
                                # start and stop times
                                trialStartIdx = self.behdict['trialStartTimes'][0][triali],
                                trialEndIdx   = self.behdict['trialEndTimes'][0][triali],

                                # count data for licks
                                LickCountsL = self.behdict['lickNumL'][0][triali], 
                                LickCountsR = self.behdict['lickNumR'][0][triali],

                                # boolean behavioral variables
                                irrelevantLR  = self.behdict['irrelLRs'][0][triali],
                                relevantLR    = self.behdict['trialLRs'][0][triali],
                                setShiftingID = self.behdict['setIDs'][0][triali],
                                choiceOutcome = self.behdict['trialCorrect'][0][triali],
                                trialRewarded = trialRewardIdx[triali],

                                # duration variables
                                trialDuration=self.behdict['trialSampleLength'][0][triali]/srate,
                                stimDuration=stim_duration[triali],

                                # --indexing variables-- #

                                # trial reward times
                                trialRewardTimesIdx=trialRewardTimeIdx[triali],
                                trialRewardTimes=trialRewardTime[triali],

                                # stimulus onset
                                stimulusOnsetIdx  = self.behdict['stimOnTimes'][0][triali],
                                stimulusOnsetTime = timestamps[self.behdict['stimOnTimes'][0][triali]],

                                # stimulus offset
                                stimulusOffsetIdx  = self.behdict['stimOffTimes'][0][triali],
                                stimulusOffsetTime = timestamps[self.behdict['stimOffTimes'][0][triali]],                              

                                # timeseries data to reference against
                                timeseries=[time_series])
                
            '''

            stimOnInterval = TimeIntervals(
                name="stim_on",
                description="Intervals aligned to stimulus onset",
            )
            stimOffInterval = TimeIntervals(
                name="stim_off",
                description="Intervals aligned to stimulus offset",
            )            
            rewTimeInterval = TimeIntervals(
                name="rew_times",
                description="Intervals aligned to reward delivery",
            )   
            rewTimeCorInterval = TimeIntervals(
                name="rew_times_correct",
                description="Intervals aligned to reward delivery on correct trials",
            )   
            rewTimeIncorInterval = TimeIntervals(
                name="rew_times_incorrect",
                description="Intervals aligned to reward delivery on incorrect trials",
            )   
            '''
            lickLInterval = TimeIntervals(
                name="lickL_times",
                description="Intervals aligned to lick left times",
            )  
            lickRInterval = TimeIntervals(
                name="lickR_times",
                description="Intervals aligned to lick right times",
            )  
            '''
            for triali in range(len(self.behdict['trialNum'][0])):
                stimOnInterval.add_row(start_time  = timestamps[self.behdict['stimOnTimes'][0][triali]],
                                       stop_time   = timestamps[self.behdict['stimOffTimes'][0][triali]])  
                stimOffInterval.add_row(start_time = timestamps[self.behdict['stimOffTimes'][0][triali]],
                                        stop_time  = timestamps[self.behdict['stimOffTimes'][0][triali]])  
            '''

            # The variables below do not follow trial structure

            #TODO: I don't know if we need this below. This is just tabular form.
            print("Please wait, this may take a few moments...")
            '''
            for evi in range(len(self.behdict['rewardTimes'][0])):                
                rewTimeInterval.add_row(start_time = timestamps[self.behdict['rewardTimes'][0][evi]],
                                        stop_time  = timestamps[self.behdict['rewardTimes'][0][evi]])

            for evi in range(len(trialRewardCorrectTimes)):                
                rewTimeCorInterval.add_row(start_time = timestamps[trialRewardCorrectTimes[evi]],
                                       stop_time      = timestamps[trialRewardCorrectTimes[evi]])

            for evi in range(len(trialRewardIncorrectTimes)):                
                rewTimeIncorInterval.add_row(start_time = timestamps[trialRewardIncorrectTimes[evi]],
                                             stop_time  = timestamps[trialRewardIncorrectTimes[evi]])
            '''

            for evi in range(len(self.behdict['lickTimesL'][0])):
                lickLInterval.add_row(start_time  = timestamps[self.behdict['lickTimesL'][0][evi]],
                                    stop_time     = timestamps[self.behdict['lickTimesL'][0][evi]])  
            
            for evi in range(len(self.behdict['lickTimesR'][0])):
                lickRInterval.add_row(start_time  = timestamps[self.behdict['lickTimesR'][0][evi]],
                                    stop_time     = timestamps[self.behdict['lickTimesR'][0][evi]])                                                

            '''
            nwbfile.add_time_intervals(stimOnInterval)
            nwbfile.add_time_intervals(stimOffInterval)
            nwbfile.add_time_intervals(rewTimeInterval)
            nwbfile.add_time_intervals(rewTimeCorInterval)
            nwbfile.add_time_intervals(rewTimeIncorInterval)
            '''
            nwbfile.add_time_intervals(lickLInterval)
            nwbfile.add_time_intervals(lickRInterval)            

            io.write(nwbfile)
            
        # rewrite
        #io.close()

        # need to find a way to get specific trial information! LR Odor Whisker and then loop add the data to the NWB file
        # moreover, it would be nice to add calcium transients as a part of the timeseries
        # maybe I add a dynamic table like how I did with tetrodes??

# an unwrapping class to unpackage data
class unwrap():
    def __init__(self):
        pass
    def spellmanBeh(self, nwbpath: str):

        """
        This function unwraps the nwb behavior file
        """
        behdict = dict()

        # read existing file
        io = NWBHDF5IO(nwbpath, mode="r") # read in editor mode
        nwbfile = io.read()

        # behavioral data
        nwbbeh = nwbfile.processing['behavior'].data_interfaces['BehavioralEpochs'].interval_series
        behdict['lickTimesAll'] = [nwbbeh['lick_times_all_index'].data[:]]
        behdict['lickTimesR']   = [nwbbeh['lick_times_right_index'].data[:]]
        behdict['lickTimesL']   = [nwbbeh['lick_times_left_index'].data[:]]
        behdict['trialStartTimes'] = [nwbbeh['trial_start_index'].data[:]]
        behdict['trialEndTimes'] = [nwbbeh['trial_end_index'].data[:]]
        behdict['stimOnTimes'] = [nwbbeh['stim_on_times'].data[:]]
        behdict['stimOffTimes'] = [nwbbeh['stim_off_times'].data[:]]
        behdict['rewardTimes'] = [nwbbeh['reward_time_index'].data[:]]
        
        # frameTimes
        behdict['frameTimes'] = [nwbfile.acquisition['frameTimes'].data[:]]

        # get trials data
        df = nwbfile.trials.to_dataframe()
        behdict['lickNumL'] = [np.array(df['LickCountsL'])]
        behdict['lickNumR'] = [np.array(df['LickCountsR'])]
        behdict['irrelLRs']  =[np.array(df['irrelevantLR'])]
        behdict['trialLRs']  = [np.array(df['relevantLR'])]
        behdict['setIDs']   = [np.array(df['setShiftingID'])]
        behdict['trialCorrect'] = [np.array(df['choiceOutcome'])]

        return behdict

# function to check the consistency between NWB files and the corresponding behavioral files
def checkConsistency(suite2p_path,matpath):
    """
    Checks if your suite2p data and behavioral files have the same shape in timestamps.

    Args:
        >>> suite2p_path: path to your suite2p file (not plane0)
        >>> matpath: path to your .mat file

    Returns:
        >>> matched: Boolean (True/False) telling you if your .mat and .npy files are aligned
                if True, then your fluorescence data and behavioral data are good to compare.
                if False, then your fluorescence data and behavioral data are not good to align

    John Stout
    """

    # load in the Fluorescence and behavioral timestamps to make sure their data aligns
    F_shape   = np.array(np.load(os.path.join(suite2p_path,'plane0','F.npy'), allow_pickle=True).shape)
    beh_shape = np.array(scipy.io.loadmat(matpath)['frameTimes'][0].shape)

    if F_shape[1] != beh_shape[0]:
        matched=False
    else:
        matched=True

    return matched

def validate_nwb(nwbpath: str):
        """
        Used to validate NWB file

        TODO: validate against a known file with issues.
        """
        val_out = validate(paths=[nwbpath], verbose=True)
        print("NWB validation may be incorrect. Still need an invalid NWB file to check against....10/10/2023")
        if val_out[1]==0:
            print("No errors detected in NWB file")
        else:
            print("Error detected in NWB file")
            # print a .txt file with an error
            root = os.path.split(nwbpath)[0]
            nwbf = os.path.split(nwbpath)[-1]
            txtpath = os.path.join(root,nwbf.split('.nwb')[0]+'_FILE-ERROR.txt')
            with open(txtpath, "w") as file:
                file.write("NWB validation failed. Consider removing this file.")

# load nwb file
#suite2p_nwb.read_nwb(existing_nwb_path)
#self = nwbfun.behavior_nwb(matpath = matpath)
# load behavior
#behdata = scipy.io.loadmat(matpath)
#varnames = behdata.keys()

'''
lickL_ts = TimeSeries(name="lickLcounts", data=self.behdict['lickNumL'], 
            unit="Counts", starting_time=0.0, rate=1.0)
lickR_ts = TimeSeries(name="lickRcounts", data=self.behdict['lickNumR'], 
            unit="Counts", starting_time=0.0, rate=1.0)   
irrelLR_ts = TimeSeries(name="irrelevantLR", data=self.behdict['irrelLRs'], 
            unit="Boolean", starting_time=0.0, rate=1.0)   
relLR_ts = TimeSeries(name="relevantLR", data=self.behdict['trialLRs'], 
            unit="Boolean", starting_time=0.0, rate=1.0)    
setID_ts = TimeSeries(name="setShiftingID", data=self.behdict['setIDs'], 
            unit="Boolean", starting_time=0.0, rate=1.0)                 
outcome_ts = TimeSeries(name="correct", data=self.behdict['trialCorrect'], 
            unit="Boolean", starting_time=0.0, rate=1.0)   


    
sleep_stages.add_column(name="stage", description="stage of sleep")
sleep_stages.add_column(name="confidence", description="confidence in stage (0-1)")

sleep_stages.add_row(start_time=0.3, stop_time=0.5, stage=1, confidence=0.5)
sleep_stages.add_row(start_time=0.7, stop_time=0.9, stage=2, confidence=0.99)
sleep_stages.add_row(start_time=1.3, stop_time=3.0, stage=3, confidence=0.7)

_ = nwbfile.add_time_intervals(sleep_stages)      

'''