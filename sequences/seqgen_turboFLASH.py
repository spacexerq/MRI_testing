# -*- coding: utf-8 -*-
"""
A subroutine to generate the GRE (gradient echo) pulse sequence.
Requires the params structure as input.

Need to think of the reequired output (pulse sequence variable itself?)

@author: m13slash9
"""

import pypulseq as pp
import numpy as np

import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)

from utilities import phase_grad_utils as pgu
from utilities.magn_prep.magn_prep import magn_prep_calc as magn_prep_calc
from utilities.magn_prep.magn_prep import magn_prep_add as magn_prep_add

def seqgen_turboFLASH(param):
    
    
    # Constant in degrees for RF spoiling (180 for alternating pulsese in FISP)
    if True:
        phase_increment = param.RF_spoil
    else:
        phase_increment = 0
    
    # Read scanner parameters from the params structure
    scanner_parameters = pp.Opts(max_grad = param.G_amp_max, grad_unit = 'Hz/m',\
                                 max_slew = param.G_slew_max, slew_unit = 'Hz/m/s',\
                                    grad_raster_time = param.grad_raster_time,\
                                    rf_raster_time = param.rf_raster_time,\
                                    adc_raster_time = 1/(param.BW_pixel*param.Nf),\
                                    block_duration_raster = max(param.grad_raster_time,param.rf_raster_time))
    
    IR_module_elements, IR_module_duration = magn_prep_calc(vars(param),scanner_parameters)
        
    # For all modules
    gradient_ramp_time = param.grad_raster_time*np.ceil((scanner_parameters.max_grad/scanner_parameters.max_slew)/param.grad_raster_time)

    # Inversion module
    # pulse_bandwidth_ir = np.double(param.t_BW_product_inv) / np.double(param.t_inv) #In Hz
    # gradient_inv_slice_amp = np.double(pulse_bandwidth_ir) / np.double(param.sl_thkn) #In Hz/m
    # gradient_ir = pp.make_trapezoid(channel = 'z',
    #                             flat_time = np.double(param.t_inv), 
    #                             rise_time = gradient_ramp_time, 
    #                             amplitude = gradient_inv_slice_amp, 
    #                             system = scanner_parameters) 
    
    # # Slice-selection gradient should be manaully calculated for precise slope timing
    # inv_pulse = pp.make_sinc_pulse(flip_angle = np.radians(param.inv_flip_angle), 
    #                                   system = scanner_parameters, 
    #                                   duration = param.t_inv,
    #                                   slice_thickness = param.sl_thkn, 
    #                                   apodization = 0.3,
    #                                   time_bw_product = param.t_BW_product_inv, 
    #                                   phase_offset = np.radians(90),
    #                                   delay = gradient_ramp_time);
    # delay_IR_time = np.ceil((param.TI - 0.5*param.t_inv - 2.0*gradient_ramp_time - 0.5*np.double(param.t_ex) - (np.floor(param.Np*0.5) + param.D_scans)*param.TR - param.TE) / scanner_parameters.grad_raster_time) * scanner_parameters.grad_raster_time
    # delay_IR = pp.make_delay(delay_IR_time)
    # Spoiling gradient must be added
        
    # Calculate slice-selection gradient
    pulse_bandwidth = np.double(param.t_BW_product_ex) / np.double(param.t_ex) #In Hz
    gradient_slice_amp = np.double(pulse_bandwidth) / np.double(param.sl_thkn) #In Hz/m
    gradient_slice = pp.make_trapezoid(channel = 'z',
                                flat_time = np.double(param.t_ex), 
                                rise_time = gradient_ramp_time, 
                                amplitude = gradient_slice_amp, 
                                system = scanner_parameters)
    # Generate the RF pulse
    exc_pulse_offsets = (np.linspace(0.0,param.sl_nb - 1.0,np.int16(param.sl_nb)) - 0.5*(np.double(param.sl_nb)  - 1.0))*(param.sl_thkn*(100.0 + param.sl_gap)/100.0)*gradient_slice_amp
    exc_pulse = pp.make_sinc_pulse(flip_angle = np.radians(param.FA),\
                                duration = np.double(param.t_ex),\
                                #freq_offset = curr_offset,\
                                phase_offset = 0,\
                                time_bw_product = param.t_BW_product_ex,\
                                delay = gradient_ramp_time,\
                                system = scanner_parameters)
    
    # Align RF and slice gradient
    #all_excit_aligned = pp.align(center = exc_pulses + [gradient_slice])     
      
    # Calculate the readout k-space span
    k_span_RO = (np.double(param.Nf)/np.double(param.FoV_f)) #In 1/m
    
    # Max area without flat part
    max_blip_area = scanner_parameters.max_grad*gradient_ramp_time #In 1/m
    
    # Min frequency prewinder duration
    read_grad_ramp_area = 0.5*gradient_ramp_time*(param.BW_pixel*param.Nf)/param.FoV_f #In 1/m
    ro_prew_area = 0.5*k_span_RO + read_grad_ramp_area
    if ro_prew_area <= max_blip_area:
        min_ro_prew_time = 2.0*gradient_ramp_time #In s
    else:
        min_ro_prew_time =  2.0*gradient_ramp_time + (ro_prew_area-max_blip_area)/scanner_parameters.max_grad #In s
    
    # Min slice-selection rewinder duration
    #All thigs here should include pulse phase centering
    if 0.5*gradient_slice.area <= max_blip_area:
        min_ss_rewi_time = 2.0*gradient_ramp_time #In s
    else:
        min_ss_rewi_time = 2.0*gradient_ramp_time + (0.5*gradient_slice.area-max_blip_area)/scanner_parameters.max_grad #In s, Pulse centring should go here
    
    # Min phase gradient duration
    k_span_PE = np.double(param.Np)/np.double(param.FoV_p)
    k_steps_PE = pgu.create_k_steps(k_span_PE, np.int16(param.Np))
    max_pe_area = np.max(np.abs(k_steps_PE))
    
    # Sort for the correcct effective echo placement
    #required_shift = np.int16(param.N_TI - np.int16( np.floor(np.double(param.Np)*0.5) + 1.0 ))
    #k_steps_PE = np.roll(k_steps_PE, required_shift)
    
    # Append dummy scans
    k_steps_PE = np.array([k_steps_PE[0] for x in range(param.D_scans)] + k_steps_PE.tolist())
    if max_pe_area <= max_blip_area:
        min_pe_time = 2.0*gradient_ramp_time
    else:
        min_pe_time = 2.0*gradient_ramp_time + (max_pe_area-max_blip_area)/scanner_parameters.max_grad #In s
    # Get the maximum of all these timings
    pe_duration = np.max(np.array([min_ro_prew_time,min_ss_rewi_time,min_pe_time]))
    pe_duration = param.grad_raster_time*np.ceil(pe_duration/param.grad_raster_time)
	
    # To have at least one count of flat area
    if pe_duration == 2.0*gradient_ramp_time:
        pe_duration = pe_duration + param.grad_raster_time
    
    # Min slice (spoiler?) duration
    slice_spoil_area = param.spoil_strenght*1.0/param.sl_thkn
    if slice_spoil_area <= max_blip_area:
        min_ss_spoil_time = 2.0*gradient_ramp_time #In s
    else:
        min_ss_spoil_time =  2.0*gradient_ramp_time + (slice_spoil_area-max_blip_area)/scanner_parameters.max_grad #In s
    slice_spoil_area_list = slice_spoil_area - 0.1*slice_spoil_area*np.random.rand(k_steps_PE.size)
    
    # Min read (spoiler?) duration
    read_spoil_area = param.spoil_strenght*k_span_RO #Should be 1.0*k_span_RO
    if read_spoil_area <= max_blip_area:
        min_ro_spoil_time = 2.0*gradient_ramp_time #In s
    else:
        min_ro_spoil_time =  2.0*gradient_ramp_time + (read_spoil_area-max_blip_area)/scanner_parameters.max_grad #In s    
    read_spoil_area_list = read_spoil_area - 0.1*read_spoil_area*np.random.rand(k_steps_PE.size)
    
    # Get the maximum of all these timings
    pe_rew_duration = np.max(np.array([min_ss_spoil_time,min_ro_spoil_time]))
    pe_rew_duration = param.grad_raster_time*np.ceil(pe_rew_duration/param.grad_raster_time)
    
    # To have at least one count of flat area
    if pe_rew_duration == 2.0*gradient_ramp_time:
        pe_rew_duration = pe_rew_duration + param.grad_raster_time
    
    # Generate the slice rewinder
    gradient_slice_rewind = pp.make_trapezoid(channel = 'z',
                                duration = np.double(pe_duration), 
                                #flat_time = np.double(pe_duration), 
                                rise_time = gradient_ramp_time,
                                area = -0.5*gradient_slice.area,
                                #amplitude = gradient_slice_amp, 
                                system = scanner_parameters)
    
    # Generate the phase encoder
    gradient_phase_winder = list()
    for phase_step in range(k_steps_PE.size):
        gradient_phase_winder.append( pp.make_trapezoid(channel = 'y',
                                duration = np.double(pe_duration), 
                                #flat_time = np.double(pe_duration), 
                                rise_time = gradient_ramp_time,
                                area = k_steps_PE[phase_step],
                                #amplitude = gradient_slice_amp, 
                                system = scanner_parameters) )

    # Generate the readout prewinder
    gradient_read_prewinder = pp.make_trapezoid(channel = 'x',
                                duration = np.double(pe_duration), 
                                #flat_time = np.double(pe_duration), 
                                rise_time = gradient_ramp_time,
                                area = -k_span_RO*0.5 - read_grad_ramp_area,
                                #amplitude = gradient_slice_amp, 
                                system = scanner_parameters)
    
    # Calculate the TE pause time
    te_pause = param.TE - (2*gradient_ramp_time + np.double(param.t_ex)*0.5 + pe_duration + 0.5/param.BW_pixel)
    te_pause = max(0.0,te_pause)
    te_pause = max(param.grad_raster_time,param.rf_raster_time)*np.floor(te_pause/max(param.grad_raster_time,param.rf_raster_time))
    
    # Generate the TE delay
    te_filler = pp.make_delay(te_pause)
    
    # Generate the readout gradient
    gradient_read = pp.make_trapezoid(channel = 'x',
                                #duration = np.double(pe_duration), 
                                flat_time = np.double(1/param.BW_pixel), 
                                rise_time = gradient_ramp_time,
                                flat_area = k_span_RO,
                                #amplitude = gradient_slice_amp, 
                                system = scanner_parameters)
    
    # Generate the ADC event
    adc_module = pp.make_adc(num_samples = param.Nf, duration = 1/param.BW_pixel, delay = gradient_ramp_time, system = scanner_parameters)
    
    # Generate the spoilers
    gradient_slice_spoil_list = list()
    gradient_read_spoil_list = list()  
    for spoil_area_s, spoil_area_r in zip(slice_spoil_area_list,read_spoil_area_list):
        gradient_slice_spoil = pp.make_trapezoid(channel = 'z',
                                duration = np.double(pe_rew_duration), 
                                #flat_time = np.double(pe_duration), 
                                rise_time = gradient_ramp_time,
                                area = spoil_area_s,
                                #amplitude = gradient_slice_amp, 
                                system = scanner_parameters)
        gradient_slice_spoil_list.append(gradient_slice_spoil)
      
        gradient_read_spoil = pp.make_trapezoid(channel = 'x',
                                duration = np.double(pe_rew_duration), 
                                #flat_time = np.double(pe_duration), 
                                rise_time = gradient_ramp_time,
                                area = spoil_area_r,
                                #amplitude = gradient_slice_amp, 
                                system = scanner_parameters)
        gradient_read_spoil_list.append(gradient_read_spoil)
    
    # Calculate the multislice loop properties
    useful_duration = 2.0*gradient_ramp_time + param.t_ex + pe_duration + te_pause + \
                        2.0*gradient_ramp_time + 1/param.BW_pixel + pe_rew_duration
    #slices_per_TR = np.floor(param.TR/useful_duration)       
    #required_concats = np.int32(np.ceil(param.sl_nb/slices_per_TR))
    slice_list = list(range(np.int32(param.sl_nb)))
    #slice_list = [slice_list[x::required_concats] for x in range(required_concats)]
    
    # Calculate the TR fillers
    tr_pause = param.TR - useful_duration
    tr_pause = max(param.grad_raster_time,param.rf_raster_time)*np.floor(tr_pause/max(param.grad_raster_time,param.rf_raster_time))
    
    # Generate the TR fillers
    tr_filler = pp.make_delay(tr_pause)
    
    #-------- Construct pulse sequence --------
    # Initialize the sequence
    gre_sequence = pp.Sequence(system = scanner_parameters)
    print(slice_list)
    for curr_slice in slice_list:
        # Add inversion
        #gre_sequence.add_block(inv_pulse, gradient_ir)
        #gre_sequence.add_block(delay_IR)
        gre_sequence = magn_prep_add(IR_module_elements,gre_sequence)
        phase_shift = 0
        dummy_counter = 0.5
        for phase_grad_cntr,slice_spoil_cntr,read_spoil_cntr in zip(gradient_phase_winder,gradient_slice_spoil_list,gradient_read_spoil_list):
            # Add RF and gradient block
            exc_pulse.freq_offset = exc_pulse_offsets[curr_slice]
            exc_pulse.phase_offset = (phase_shift / 180) * np.pi
            gre_sequence.add_block(exc_pulse, gradient_slice)
            # Add the prewinder/slice rewinder/phase encoder
            gre_sequence.add_block(gradient_slice_rewind, phase_grad_cntr, gradient_read_prewinder)
            # Add the TE filler
            gre_sequence.add_block(te_filler)
            # Add the ADC and readout gradient
            adc_module.phase_offset = (phase_shift / 180) * np.pi
            if dummy_counter > (param.D_scans):
                gre_sequence.add_block(adc_module, gradient_read)
            else:
                gre_sequence.add_block(gradient_read)
                dummy_counter = dummy_counter + 1.0
            # Add the spoilers and phase rewind
            gre_sequence.add_block(slice_spoil_cntr, read_spoil_cntr)
            # Add the TR filler
            gre_sequence.add_block(tr_filler)
            phase_shift = divmod(phase_shift + phase_increment,360)[-1]
            
    print(pp.check_timing.check_timing(gre_sequence)) 
    return gre_sequence