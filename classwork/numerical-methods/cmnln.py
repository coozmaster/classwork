#!/bin/python -B
import sys
import numpy as np
import scipy.optimize as scopt
import re
import matplotlib.pyplot as plt
import argparse
import os
import warnings
import yaml

sys.dont_write_bytecode = True
###############################################################################
def main():

    intro('Compressor Meanline','v0.4','2/12/2016')

    args = get_args()

    regurgitate(args)

    consts_and_conversion_factors()

    model = get_user_input(args.fn)

    model = get_calc_locations(model)

    model = meanline_solve(model)

    model = additional_calculations(model)

    model = create_help(model)

    model = create_units(model)

    output(args,model)

    if( args.debug ):

        out = model
        model = None
        global model
        model = out

        return(model)

    else:
        return()

###############################################################################
def intro(s1,s2,s3):
    '''
    print name of program, version, and date to the user
    '''
    print("\n\n",end='',sep='')
    print("+","-"*78,"+",end="\n",sep='')
    print("|",str(s1).center(78,' '),"|",end="\n",sep='')
    print("|",str(s2+' '+s3).center(78,' '),"|",end='\n',sep='')
#    print("|",str('andio'.upper()).center(78,' '),'|',end='\n',sep='')
    print("+","-"*78,"+",end="\n\n",sep='')
    sys.stdout.flush()
    return()

def get_args():
    '''
    use argument parser to obtain user inputs from command line
    '''
    parser = argparse.ArgumentParser(description="Compressor Meanline (1D) Analysis")
    parser.add_argument('--input','-i',default=None,type=str,dest='fn',\
            help="input file name")
    parser.add_argument('--no-plots','-np',default=False,dest='plots',\
            help="suppress output of plots",action='store_true')
    parser.add_argument('--debug',default=True,dest='debug',help='debug mode')


    args = parser.parse_args()

    return(args)

def check_args(args):
    '''
    perform basic argument check on user inputs
    '''
    if( not args.fn == None):
        if( not os.path.isfile(args.fn)):
            print("Input file does not exist")
            sys.exit(1)

        elif( not os.access(args.fn,os.R_OK) ):
            print("Input file is not readable")
            sys.exit(1)

    else:
        print("Input file type is None")
        sys.exit(1)

    if( not args.plots == True and not args.plots == False):
        print("How did you even get this error message?")
        sys.exit(1)

    return()

def regurgitate(args):
    '''
    throw the user input back to user to ensure what the program thinks it recieved
    and what the user thinks they input are the same
    '''
    print("Inputs\n","-"*20,end="\n",sep='')
    print("{0:20s}: {1:<30s}".format("Input file name",str(args.fn)),end="\n",sep='')
    print("{0:20s}: {1:<30s}".format("Plotting",str(not args.plots)),end="\n",sep='')
    print("\n\n",end='',sep='')

#    print("Output\n","-"*20,end="\n",sep='')
    check_args(args)

def consts_and_conversion_factors():
    '''
    define constants and conversion factors
    '''
    # conversion factors
    global in2ft,ft2in,deg2rad,rad2deg,rpm2rad_s,rad_s2rpm,HP2lbfft_s,lbfft_s2HP
    global gc,degF2degR,degR2degF,Tref,Pref,R,tol

    in2ft      = np.float64(1./12.)
    ft2in      = 1./in2ft
    deg2rad    = np.pi/180.
    rad2deg    = 180./np.pi
    rpm2rad_s  = 2*np.pi / 60.
    rad_s2rpm  = 1./rpm2rad_s
    HP2lbfft_s = np.float64(550.)
    lbfft_s2HP = 1/HP2lbfft_s
    gc         = np.float64(32.174)
    degF2degR  = np.float64(459.67)
    degR2degF  = -1. * degF2degR
    Tref       = np.float64(59. + degF2degR)
    Pref       = np.float64(14.696 * 144.)
    R          = np.float64(53.34) * gc
    tol        = np.float64(10**(-6))

    return()


def get_user_input(fn):
    '''
    get user input values from input YAML file
    '''
    with open(fn,'r') as f:
        model = yaml.load(f)

    fn_base = fn.split('.')[0]
    model['base'] = fn_base

    model = error_check_yaml(model)

    for sta in model['stations']:

        # convert values to numpy 64 bit floats
        for i in model['stations'][sta]:
            if( not type(model['stations'][sta][i]) == str and \
                not type(model['stations'][sta][i]) == dict ):
                model['stations'][sta][i] = np.float64(model['stations'][sta][i])
            else:
                next

        for j in ['xinner','xouter','rinner','router']:
            model['stations'][sta][j] = model['stations'][sta][j] * in2ft

        for j in ['Q']:
            model['stations'][sta][j] = model['stations'][sta][j] * HP2lbfft_s

        if( model['stations'][sta]['type'] == 'statorte'):
            for k in ['alphate']:
                model['stations'][sta][k] = model['stations'][sta][k] * deg2rad



    return(model)

def error_check_yaml(model):
    '''
    perform error checks on values read in from YAML input file
    if a flag is tripped to true when the function is complete it will
    throwback an error and exit with all error statements for the user
    this allows the user to fix all errors in single file edit
    '''
    flag = False

    for sta in model['stations']:
        keys = model['stations'][sta]

        if( not any('type' in s for s in model['stations'][sta])):
            print('Station {0:3d} has no {1:<10s} declared\n'.format(sta,'type'))
            sys.exit(1)

        for check in ['xinner','xouter','rinner','router','loss']:

            if(  not any(check in i for i in model['stations'][sta])):
                print('Station {0:3d} has no {1:<10s} declared\n'.format(sta,check))
                flag = True

        if( model['stations'][sta]['type'] == 'rotorte'):

            if( not any('PR' in i for i in model['stations'][sta])):
                if( not any('workcf' in i for  i in model['stations'][sta])):
                    print('Sta:{0:3d}| type:{1:>10s}| msg: requires PR or workcf defined\n'.format(\
                            sta,str(model['stations'][sta]['type'])))
                    flag = True

            for check in ['nbld','gear','solidity']:
                if( not any(check in i for i in model['stations'][sta])):
                    print('Station {0:3d} has no {1:<10s} declared\n'.format(sta,check))
                    flag = True

            # any unnecessary items are auto-set to zero
            for notreq in ['alphate']:
                    model['stations'][sta][notreq] = np.float64(0.)


            # if PR is set to a non-zero value then set workcf to 0 and vice versa
            if( not model['stations'][sta]['PR'] == 0 ):
                    model['stations'][sta]['workcf'] = np.float64(0.)

            elif( not model['stations'][sta]['workcf'] == 0):
                    model['stations'][sta]['PR'] = 0


        elif( model['stations'][sta]['type'] == 'statorte'):
            for check in ['alphate','nbld','solidity','loss']:
                if( not any(check in i for i in model['stations'][sta])):
                    print('Station {0:3d} has no {1:<10s} declared\n'.format(sta,check))
                    flag = True

            # set any unnecessary values to zero
            for notreq in ['PR','gear','workcf']:
                    model['stations'][sta][notreq] = np.float64(0.)

        elif( model['stations'][sta]['type'] == 'cam'):
            for check in ['loss']:
                if( not any(check in i for i in model['stations'][sta])):
                    print('Station {0:3d} has no {1:<10s} declared\n'.format(sta,check))
                    flag = True

            # set any unnecessary values to zero
            if( not model['stations'][sta+1]['type'] == 'rotorte'):
                for notreq in ['PR','gear','workcf','solidity','nbld','alphate']:
                    model['stations'][sta][notreq] = 0.
            else:
                for notreq in ['PR','workcf','solidity','nbld','alphate']:
                    model['stations'][sta][notreq] = 0.


        # any values that are required but were omitted are assumed to be 0 (such as mbld or Q)
        for omitted_but_req in ['Q','mbld']:
            if( not any(omitted_but_req in i for i in model['stations'][sta])):
                model['stations'][sta][omitted_but_req] = np.float64(0.)

    if( flag ):
        sys.exit(1)
    return(model)

def get_calc_locations(model):
    '''
    get the calculation locations (RMS radius) based on user input axial (x)
    and user input radial (r) locations
    '''
# determine root mean square locations (which splits annular area into equal parts)
# these are the locations where the equations will be used
    model['xinner']   = np.zeros([len(model['stations'])])

    names = ['xinner','rinner','xouter','router','xrms','rrms','dxinner','dxouter',\
             'drouter','dxrms','drrms','dsinner','dsouter','solidity','gear',\
             'drinner','PR','alphate','nbld','loss','Q','B',\
             'workcfrms','workcftip','workcfhub']

    for name in names:
        model[name] = np.zeros_like(model['xinner'])



    for sta in model['stations']:
        for name in names:
            if( any(name in i for i in model['stations'][sta])):
                model[name][sta] = model['stations'][sta][name]

            else:
                model['stations'][sta][name] = model[name][sta]

    # begin array operations
    model['dxinner'][1:] = np.diff(model['xinner'])
    model['dxouter'][1:] - np.diff(model['xouter'])
    model['drinner'][1:] = np.diff(model['rinner'])
    model['drouter'][1:] = np.diff(model['router'])

    model['dsinner'] = np.sqrt( model['dxinner']**2. + model['drinner']**2.)
    model['dsouter'] = np.sqrt( model['dxouter']**2. + model['drouter']**2.)
    model['dx'] = model['xouter'] - model['xinner']
    model['dr'] = model['router'] - model['rinner']
    model['rrms'] = np.sqrt(0.5*(model['router']**2. + model['rinner']**2.))
    model['xrms'] = model['dx']/model['dr'] \
                    * (model['rrms'] - model['rinner']) + model['xinner']

    model['drrms'][1:] = np.diff(model['rrms'])
    model['dxrms'][1:] = np.diff(model['xrms'])
    model['dsrms']     = np.sqrt( model['drrms']**2. + model['dxrms']**2.)

    # now put the array information back into the station information
    for sta in model['stations']:
        for name in ['dxinner','dxouter','drinner','drouter','dsinner','dx',\
                     'dsouter','dr','rrms','xrms','drrms','dxrms','dsrms']:

            model['stations'][sta][name] = model[name][sta]

    return(model)

def setup_bleeds(model):
    '''
    take user bleed information and setup mass flow, total pressure,
    and total temperature for use in solving the conservation equations
    in meanline_solve
    '''
    model['mbld'] = np.zeros_like(model['xinner'])
    model['Poinj'] = np.zeros_like(model['xinner'])
    model['Toinj'] = np.zeros_like(model['xinner'])

    for sta in model['stations'].keys():

        if( model['stations'][sta]['mbld'] == 0.):
            next

        elif( isinstance(model['stations'][sta]['mbld'],dict)):

            if( re.search('(?ix) inj(ection)?',model['stations'][sta]['mbld']['type'])):
                model['mbld'][sta] = np.float64(\
                    model['stations'][sta]['mbld']['flowfraction'])
                model['Poinj'][sta] = np.float64(\
                    model['stations'][sta]['mbld']['Poinj'])
                model['Toinj'][sta] = np.float64(\
                    model['stations'][sta]['mbld']['Toinj'])

            elif( re.search('(?ix) ext(raction)?',model['stations'][sta]['mbld']['type'])):

                model['mbld'][sta] = np.float64(\
                    -1. * model['stations'][sta]['mbld']['flowfraction'])

    return(model)

def meanline_solve(model):
    '''
    proceed station by station sequentially through the machine setting
    appropriate conditions based on user inputs and solve energy, continuity,
    and angular momentum
    '''
    names = ['Vx','Vr','Vm','Vu','Vurel','V','Vrel','Angmom','A','phi','Po',\
             'Porel','To','Torel','ho','horel','rothalpy','po','theta','delta',\
             'P','Pd','Pdrel','T','p','R','k','Nm','ma','Nc','alpha','beta',\
             'Mx','Mr','Mm','Mu','Murel','M','Mrel','Torque','Poloss',\
             'omega','flowcfrms','flowcftip','flowcfhub','psi','chi','xi']

    for name in names:

        model[name] = np.zeros_like(model['xinner'])

    if( model['options']['gasmodel'].lower() == 'ideal'):
            model['k']  = np.ones_like(model['xinner']) * np.float64(1.4)
            model['R']  = np.ones_like(model['xinner']) * R
            model['cp'] = model['k']*model['R']/(model['k']-1.)
            model['cv'] = model['R']/(model['k']-1.)

    elif( model['gasmodel'].lower() == 'real'):
            print('real gas - placeholder')
            sys.exit(1)

    # get angle between station line and engine CL
    model['psi'] = np.arctan2(model['dr'],model['dx'])

    # chi is the angle of each individual rms line and the engine CL between stations
    model['chi'] = np.arctan2(model['drrms'],model['dxrms'])

    # flow angle is estimated from average of up and downstream station chi values
    # the last value of phi is linearly extrapolated
    if( model['inlet']['phi'].lower() == 'station'):
        model['phi'][0] = model['chi'][0]
    else:
        model['phi'][0] = model['inlet']['phi']

    model['phi'][1:-1] = 0.5 * (model['chi'][1:-1] + model['chi'][2:])
    model['phi'][-1]   = model['chi'][-1]

    # begin area calculation
    model['A'] = np.pi * (model['router']**2. - model['rinner']**2.) * \
                   np.sin(model['psi']-model['phi'])/np.sin(model['psi'])

    # setup bleeds (injections and extractions)
    model = setup_bleeds(model)

    # begin iteration from station to station
    for sta in range(len(model['xinner'])):
        if( sta == 0):

            model['inlet']['theta'] = model['inlet']['To'] / Tref
            model['inlet']['delta'] = model['inlet']['Po'] / Pref

            model['theta'][sta]     = get_theta(model,sta)
            model['delta'][sta]     = get_delta(model,sta)

            model['inlet']['Nm']    = model['inlet']['Nc'] * model['theta'][sta]
            model['inlet']['ma']    = model['inlet']['mc'] * model['delta'][sta]\
                / model['theta'][sta]

            model['ma'][sta]        = get_ma(model,sta)

            model['alpha'][sta]     = get_alpha(model,sta)
            model['Po'][sta]        = get_Po(model,sta)
            model['To'][sta]        = get_To(model,sta)

            model['po'][sta]        = get_po(model,sta)

            model['B'][sta]         = get_blockage(model,sta)

            model['Mm'][sta]        = get_Mm(model,sta)
            model['Mx'][sta]        = get_Mx(model,sta)
            model['Mr'][sta]        = get_Mr(model,sta)
            model['Mu'][sta]        = get_Mu(model,sta)
            model['M'][sta]         = get_M(model,sta)

            model['T'][sta]         = get_T(model,sta)

            model['P'][sta]         = get_P(model,sta)
            model['Pd'][sta]        = get_Pd(model,sta)

            model['Vm'][sta]        = get_Vm(model,sta)
            model['Vx'][sta]        = get_Vx(model,sta)
            model['Vr'][sta]        = get_Vr(model,sta)
            model['Vu'][sta]        = get_Vu(model,sta)
            model['V'][sta]         = get_V(model,sta)

            model['Angmom'][sta]    = get_Angmom(model,sta)

            # this is not a typo - omega is calculated for all stations simultaneously
            model['omega']          = get_omega(model)
            model['Urms'],model['Utip'],model['Uhub'] = get_U(model)
            model['rothalpy'][sta]  = get_rothalpy(model,sta)

            model['Vurel'][sta]     = get_Vurel(model,sta)
            model['Murel'][sta]     = get_Murel(model,sta)

            model['beta'][sta]      = get_beta(model,sta)
            model['Mrel'][sta]      = get_Mrel(model,sta)
            model['Vrel'][sta]      = get_Vrel(model,sta)
            model['Porel'][sta]     = get_Porel(model,sta)
            model['Pdrel'][sta]     = get_Pdrel(model,sta)

            check_ma(model,sta)

        elif( model['stations'][sta]['type'] == 'cam' ):

            model['Po'][sta]    = get_Po(model,sta)
            model['ma'][sta]    = get_ma(model,sta)
            model['To'][sta]    = get_To(model,sta)
            model['po'][sta]    = get_po(model,sta)
            model['theta'][sta] = get_theta(model,sta)
            model['delta'][sta] = get_delta(model,sta)
            model['B'][sta] = get_blockage(model,sta)

            model['Vu'][sta]    = get_Vu(model,sta)
            model['Angmom'][sta]= get_Angmom(model,sta)

            model['Mm'][sta],model['Mu'][sta] = get_Mm(model,sta)

            model['alpha'][sta] = get_alpha(model,sta)
            model['M'][sta]     = get_M(model,sta)

            model['T'][sta]     = get_T(model,sta)
            model['P'][sta]     = get_P(model,sta)
            model['Pd'][sta]    = get_Pd(model,sta)
            model['p'][sta]     = get_p(model,sta)

            model['Vm'][sta]    = get_Vm(model,sta)
            model['Vx'][sta]    = get_Vx(model,sta)
            model['Mx'][sta]    = get_Mx(model,sta)
            model['Vr'][sta]    = get_Vr(model,sta)
            model['Mr'][sta]    = get_Mr(model,sta)
            model['V'][sta]     = get_V(model,sta)

            model['Angmom'][sta]= get_Angmom(model,sta)
            model['rothalpy'][sta] = get_rothalpy(model,sta)
            model['Torque'][sta]= get_Torque(model,sta)

            model['Vurel'][sta] = get_Vurel(model,sta)
            model['Murel'][sta] = get_Murel(model,sta)
            model['beta'][sta]  = get_beta(model,sta)
            model['Mrel'][sta]  = get_Mrel(model,sta)
            model['Vrel'][sta]  = get_Vrel(model,sta)
            model['Porel'][sta] = get_Porel(model,sta)
            model['Pdrel'][sta] = get_Pdrel(model,sta)
            model['Poloss'][sta]= get_Poloss(model,sta)

            check_ma(model,sta)

        elif( model['stations'][sta]['type'] == 'statorte' ):

            model['Po'][sta]    = get_Po(model,sta)
            model['ma'][sta]    = get_ma(model,sta)
            model['To'][sta]    = get_To(model,sta)
            model['po'][sta]    = get_po(model,sta)
            model['theta'][sta] = get_theta(model,sta)
            model['delta'][sta] = get_delta(model,sta)
            model['alpha'][sta] = get_alpha(model,sta)
            model['B'][sta]     = get_blockage(model,sta)

            model['Mm'][sta]    = get_Mm(model,sta)
            model['Mu'][sta]    = get_Mu(model,sta)
            model['M'][sta]     = get_M(model,sta)
            model['T'][sta]     = get_T(model,sta)
            model['P'][sta]     = get_P(model,sta)
            model['Pd'][sta]    = get_Pd(model,sta)
            model['p'][sta]     = get_p(model,sta)
            model['Vu'][sta]    = get_Vu(model,sta)

            model['Vm'][sta]    = get_Vm(model,sta)
            model['Vx'][sta]    = get_Vx(model,sta)
            model['Mx'][sta]    = get_Mx(model,sta)
            model['Vr'][sta]    = get_Vr(model,sta)
            model['Mr'][sta]    = get_Mr(model,sta)
            model['V'][sta]     = get_V(model,sta)

            model['Angmom'][sta]= get_Angmom(model,sta)
            model['rothalpy'][sta] = get_rothalpy(model,sta)

            model['Vurel'][sta] = get_Vurel(model,sta)
            model['Murel'][sta] = get_Murel(model,sta)
            model['beta'][sta]  = get_beta(model,sta)
            model['Mrel'][sta]  = get_Mrel(model,sta)
            model['Vrel'][sta]  = get_Vrel(model,sta)
            model['Porel'][sta] = get_Porel(model,sta)
            model['Pdrel'][sta] = get_Pdrel(model,sta)
            model['Poloss'][sta]= get_Poloss(model,sta)

            model['Torque'][sta]= get_Torque(model,sta)

            check_ma(model,sta)

        elif( model['stations'][sta]['type'] == 'rotorte' ):

              check_omega(model,sta)

              model['Porel'][sta] = get_Porel(model,sta)
              model['Po'][sta]    = get_Po(model,sta)
              model['ma'][sta]    = get_ma(model,sta)

              # Entropy generation is frame invariant
              model['To'][sta]    = get_To(model,sta)
              model['rothalpy'][sta] = get_rothalpy(model,sta)
              model['Vu'][sta]    = get_Vu(model,sta)

              model['po'][sta] = get_po(model,sta)
              model['theta'][sta] = get_theta(model,sta)
              model['delta'][sta] = get_delta(model,sta)

              model['B'][sta] = get_blockage(model,sta)

              model['Mm'][sta],model['Mu'][sta] = get_Mm(model,sta)
              model['alpha'][sta] = get_alpha(model,sta)
              model['M'][sta] = get_M(model,sta)

              model['T'][sta] = get_T(model,sta)
              model['P'][sta] = get_P(model,sta)
              model['Pd'][sta] = get_Pd(model,sta)
              model['p'][sta] = get_p(model,sta)

              model['Vm'][sta] = get_Vm(model,sta)
              model['Vx'][sta] = get_Vx(model,sta)
              model['Mx'][sta] = get_Mx(model,sta)
              model['Vr'][sta] = get_Vr(model,sta)
              model['Mr'][sta] = get_Mr(model,sta)

              model['Vurel'][sta] = get_Vurel(model,sta)
              model['Murel'][sta] = get_Murel(model,sta)
              model['beta'][sta]  = get_beta(model,sta)
              model['Mrel'][sta]  = get_Mrel(model,sta)
              model['Vrel'][sta]  = get_Vrel(model,sta)

              model['Pdrel'][sta] = get_Pdrel(model,sta)
              model['V'][sta]     = get_V(model,sta)

              model['Angmom'][sta]= get_Angmom(model,sta)

              check_ma(model,sta)

              model['Torque'][sta] = get_Torque(model,sta)
              model['workcfrms'][sta],\
                  model['workcftip'][sta],\
                  model['workcfhub'][sta] = get_workcf(model,sta)
              model['flowcfrms'][sta],\
                  model['flowcftip'][sta],\
                  model['flowcfhub'][sta] = get_flowcf(model,sta)

    return(model)

def get_workcf(model,sta):
    '''
    calculate the work coefficient at the tip, rms radius, and hub
    workcf = delta total enthalpy / circumferential speed of rotor squared
    '''
    if( model['stations'][sta]['type'] == 'rotorte'):

        dho = model['cp'][sta] * model['To'][sta] - model['cp'][sta-1] * model['To'][sta-1]

        workcfrms = dho / model['Urms'][sta]**2.
        workcftip = dho / model['Utip'][sta]**2.
        workcfhub = dho / model['Uhub'][sta]**2.

    return(workcfrms,workcftip,workcfhub)

def get_flowcf(model,sta):
    '''
    claculate the flow coefficient at the tip, rms radiu,s and hub
    flowcf = meridional velocity / circumferential speed of rotor
    '''
    if( model['stations'][sta]['type'] == 'rotorte'):

        flowcfrms = model['Vm'][sta]/model['Urms'][sta]
        flowcftip = model['Vm'][sta]/model['Utip'][sta]
        flowcfhub = model['Vm'][sta]/model['Uhub'][sta]

    return(flowcfrms,flowcftip,flowcfhub)

def get_U(model):

    Urms = model['omega'] * model['rrms']
    Utip = model['omega'] * model['router']
    Uhub = model['omega'] * model['rinner']

    return(Urms,Utip,Uhub)
def check_omega(model,sta):

    if( model['stations'][sta]['type'] == 'rotorte' \
        and not model['omega'][sta] == model['omega'][sta-1]):
        print('Stations {0:3d} and {1:3d} omega (rot. rate) do not agree'.format(\
            sta,sta-1))
        sys.exit(1)

    return()

def get_Torque(model,sta):

    Torque = model['Angmom'][sta] - model['Angmom'][sta-1]

    return(Torque)

def get_Poloss(model,sta):

    Poloss = (model['Po'][sta-1] - model['Po'][sta])/model['Po'][sta-1]

    return(Poloss)

def get_mc(model,sta):

    if( sta == 0 ):
        mc = model['inlet']['mc']

    else:
        mc = model['ma'][sta] * np.sqrt(model['theta'][sta]) \
             / model['delta'][sta]

    return(mc)

def get_delta(model,sta):

    if( sta == 0 ):
        delta = model['inlet']['delta']

    else:
        delta = model['Po'][sta] / Pref

    return(delta)

def get_theta(model,sta):

    if( sta == 0 ):
        theta = model['inlet']['theta']

    else:
        theta = model['To'][sta] / Tref

    return(theta)

def get_alpha(model,sta):

    if( sta == 0 ):
       alpha = model['inlet']['alpha'] * deg2rad

    elif( model['stations'][sta]['type'] == 'cam' ):

        alpha = np.arctan2(model['Mu'][sta],model['Mm'][sta])

    elif( model['stations'][sta]['type'] == 'statorte' ):

        alpha = model['stations'][sta]['alphate']

    elif( model['stations'][sta]['type'] == 'rotorte' ):

        alpha = np.arctan2(model['Mu'][sta],model['Mm'][sta])

    else:
        print('Unknown alpha calc at station {0:3d}'.format(sta))
        sys.exit(1)

    return(alpha)

def get_ma(model,sta):

    if( sta == 0):
        ma = model['inlet']['ma'] * (1 + model['mbld'][sta])

    else:
        ma = model['ma'][sta-1] * (1 + model['mbld'][sta])

    return(ma)

def get_To(model,sta):

    if( sta == 0):
        To = model['inlet']['To'] \
             + model['Q'][sta]/(model['cp'][sta] * model['ma'][sta]/gc)

    elif( not model['stations'][sta]['type'] == 'rotorte' ):

        To = model['To'][sta-1] \
             + model['Q'][sta]/(model['cp'][sta] * model['ma'][sta]/gc)

    else:
        PRrel = (model['Porel'][sta-1] - model['loss'][sta] * model['Pdrel'][sta-1])\
                 /model['Porel'][sta-1]

        To = (model['PR'][sta]/PRrel)**((model['k'][sta]-1)/model['k'][sta])\
              * model['To'][sta-1]


    if( model['mbld'][sta] >= 0. ):

        To = To/(1+model['mbld'][sta])\
             + model['mbld'][sta]/(1+model['mbld'][sta]) * model['Toinj'][sta]

    return(To)

def get_Po(model,sta):

    if( sta == 0 ):
        Po = model['inlet']['Po']

    elif( model['options']['gasmodel'] == 'ideal' ):

        if( isinstance(model['loss'][sta],float) \
            and not model['stations'][sta]['type'] == 'rotorte'):

            Po = model['Po'][sta-1] - model['loss'][sta] * model['Pd'][sta-1]


        elif( model['stations'][sta]['type'] == 'rotorte' ):

            Po = model['Po'][sta-1] * model['PR'][sta]

    elif( model['options']['gasmodel'] == 'real' ):
        print('real gas - placeholder')


    if( model['mbld'][sta] >= 0):

        Po = Po/(1+model['mbld'][sta]) \
             + model['mbld'][sta]\
                 /(1 + model['mbld'][sta]) * model['Poinj'][sta]

    return(Po)

def get_V(model,sta):

    V = np.sqrt( model['Vm'][sta]**2. + model['Vu'][sta]**2.)

    return(V)

def get_beta(model,sta):

    beta = np.arctan2(model['omega'][sta]*model['rrms'][sta] - model['Vu'][sta],\
        model['Vm'][sta])

    return(beta)
def get_Mrel(model,sta):

    Mrel = np.sqrt(model['Murel'][sta]**2. + model['Mm'][sta]**2.)

    return(Mrel)
def get_Vrel(model,sta):

    Vrel = np.sqrt(model['Vurel'][sta]**2. + model['Vm'][sta]**2.)

    return(Vrel)

def get_Porel(model,sta):


    if( model['stations'][sta]['type'] == 'rotorte' ):
        Porel = model['Porel'][sta-1] \
                - model['loss'][sta] * model['Pdrel'][sta-1]

    else:
        Porel = model['P'][sta] * (1 + (model['k'][sta]-1)/2 * model['Mrel'][sta]**2.) \
            ** ( model['k'][sta]/(model['k'][sta]-1))

    return(Porel)

def get_Pdrel(model,sta):

    Pdrel = model['Porel'][sta] - model['P'][sta]

    return(Pdrel)
def get_Murel(model,sta):

    Murel = model['Vurel'][sta] / np.sqrt(model['k'][sta] * model['R'][sta] \
        * model['T'][sta])

    return(Murel)

def get_Vurel(model,sta):

    Vurel = model['Vu'][sta] - model['omega'][sta] * model['rrms'][sta]

    return(Vurel)

def get_omega(model):

    omega = model['inlet']['Nm'] * rpm2rad_s * model['gear']

    return(omega)

def get_rothalpy(model,sta):

    if( not model['stations'][sta]['type'] == 'rotorte'):
        rothalpy = model['cp'][sta] * model['To'][sta] \
                   - model['omega'][sta] * model['rrms'][sta] * model['Vu'][sta] \
                   + model['Q'][sta]/(model['ma'][sta]/gc)

    else:
        rothalpy = model['rothalpy'][sta-1] \
                   + model['Q'][sta]/(model['ma'][sta]/gc)

    return(rothalpy)

def get_po(model,sta):

    if( model['options']['gasmodel'].lower() == 'ideal'):
        po = model['Po'][sta]/(model['R'][sta] * model['To'][sta])

    elif( model['options']['gasmodel'].lower() == 'real'):
        print('real gas model - placeholder')

    return(po)

def get_M(model,sta):

    if( not model['Mm'][sta] == 0 and not model['Mu'][sta] == 0):

        M = np.sqrt(model['Mm'][sta]**2. + model['Mu'][sta]**2.)

    elif( not model['Mm'][sta] == 0 and not np.isnan(model['alpha'][sta]) \
        and not np.isinf(model['alpha'][sta])):

        M = model['Mm'][sta]/np.cos(model['alpha'][sta])

    elif( not model['Mu'][sta] == 0 and not np.isnan(model['alpha'][sta]) \
        and not np.isinf(model['alpha'][sta])):

        M = model['Mu'][sta]/np.sin(model['alpha'][sta])

    else:
        print('Unable to determine how to calculate absolute Mach number' \
            + ' at station {0:3d}'.format(sta))
        sys.exit(1)

    return(M)

def get_Angmom(model,sta):

    if( model['stations'][sta]['type'].lower() == 'cam'):
        if( sta == 0):
            Angmom = model['ma'][sta]/gc * model['rrms'][sta] * model['Vu'][sta]

        else:
            Angmom = model['Angmom'][sta-1]

    elif( model['stations'][sta]['type'] == 'statorte' or \
          model['stations'][sta]['type'] == 'rotorte'):

        Angmom = model['ma'][sta]/gc * model['rrms'][sta] * model['Vu'][sta]

    return(Angmom)

def get_Pd(model,sta):

    Pd = model['Po'][sta] - model['P'][sta]

    return(Pd)

def get_p(model,sta):

    if( model['options']['gasmodel'].lower() == 'ideal'):

        p = model['po'][sta] * (model['P'][sta]/model['Po'][sta])**(1/model['k'][sta])

    return(p)

def get_T(model,sta):

    if( model['options']['gasmodel'].lower() == 'ideal'):

        T = model['To'][sta] * ( 1 + (model['k'][sta]-1)/2 * model['M'][sta]**2.)**(-1.)

    return(T)

def get_P(model,sta):

    if( model['options']['gasmodel'].lower() == 'ideal'):

        P = model['Po'][sta] * (model['T'][sta]/model['To'][sta])\
            **(model['k'][sta]/(model['k'][sta]-1))


    return(P)

def get_blockage(model,sta):
    '''
    use blockage model to get value and return it to to be used in calcs later
    '''
    if( not model['options']['blockage'] == 0.0):

        if( sta > 3):
            blockage = np.float64(0.04)
        else:
            blockage = np.float64(0.01 + sta * 0.01)

    else:
        print('inside ehre yes')
        blockage = model['options']['blockage']


    return(blockage)

def get_Mm(model,sta):
    '''
    iteratively solve momentum, energy, and continuity (energy is built into continuity)
    to get value for the meriodional Mach number
    '''

    if( sta == 0):
        # call the stator variant of the function because at the inlet we forcing
        # an alpha boundary condition
        Mm_iter,infodict,ier,msg = scopt.fsolve(mass_mom_stator,(0.1),\
            args=(model['Po'][sta],model['To'][sta],model['k'][sta],model['R'][sta],\
            model['A'][sta],model['alpha'][sta],(1.0-model['B'][sta]),\
            model['ma'][sta]/gc),full_output=True)

        return(Mm_iter[0])

    elif( model['stations'][sta]['type'] == 'cam' ):
        M_iter,infodict,ier,msg = scopt.fsolve(mass_mom_cam,\
            (model['Mm'][sta-1],model['Mu'][sta-1]),\
            args=(model['Po'][sta],model['To'][sta],model['k'][sta],model['R'][sta],\
            model['A'][sta],model['rrms'][sta],(1.0-model['B'][sta]),\
            model['ma'][sta]/gc,model['Angmom'][sta]),full_output=True)

        return(M_iter[0],M_iter[1])

    elif( model['stations'][sta]['type'] == 'statorte' ):
        Mm_iter,infodict,ier,msg = scopt.fsolve(mass_mom_stator,(0.1),\
            args=(model['Po'][sta],model['To'][sta],model['k'][sta],model['R'][sta],\
            model['A'][sta],model['alpha'][sta],(1.0-model['B'][sta]),\
            model['ma'][sta]/gc),full_output=True)

        return(Mm_iter[0])

    elif( model['stations'][sta]['type'] == 'rotorte' ):
        M_iter,infodict,ier,msg = scopt.fsolve(mass_mom_rotor,\
            (model['Mm'][sta-1],model['Mu'][sta-1]),\
            args=(model['Po'][sta],model['To'][sta],model['k'][sta],\
            model['R'][sta],model['A'][sta],(1-model['B'][sta]),\
            model['ma'][sta]/gc,model['Vu'][sta]),full_output=True)

        return(M_iter[0],M_iter[1])

    if( not ier == 1):
         print(model['Po'][sta],model['To'][sta],model['k'][sta],model['R'][sta],\
            model['A'][sta],model['alpha'][sta],(1.0-model['B'][sta]),\
            model['ma'][sta]/gc)

         print('Unable to converge solution at station {0:3d}'.format(sta))

    return()

def mass_mom_stator(Mm,Po,To,k,R,A,alpha,B,ma):
    '''
    compare a calculated mass flow rate to the actual mass flow rate
    '''
    fmn  = ( 1 + (k-1)/2. * (Mm/np.cos(alpha))**2.)

    func = Po/np.sqrt(To) * np.sqrt(k/R)* A * B * Mm * fmn **(-(k+1)/(2*(k-1))) - ma

    return(func)

def mass_mom_rotor(init,Po,To,k,R,A,B,ma,Vu):
    '''
    compare calculated values of angular momentum and continuity to actual values
    '''
    Mm,Mu = init

    fmn  = ( 1 + (k-1)/2 * (Mm**2. + Mu**2.))

    func1 = Mu**2 * k * R * To * fmn**(-1) - Vu**2.

    func2 = Po/np.sqrt(To) * np.sqrt(k/R)* A * B * Mm * fmn **(-(k+1)/(2*(k-1))) - ma

    return(func1,func2)

def mass_mom_cam(init,Po,To,k,R,A,r,B,ma,Angmom):
    '''
    compare calculated values of angular momentum and mass flow rate
    '''
    Mm,Mu= init

    M  = np.sqrt( Mm**2. + Mu**2.)

    fmn = (1 + (k-1)/2 * M**2.)

    #func1 = Po * k * A * B * r * Mm * Mu * fmn **(-k/(k-1)) - Angmom
    func1 = ma * r * Mu * np.sqrt(k*R*To * fmn**(-1.)) - Angmom

    func2 = Po/np.sqrt(To) * np.sqrt(k/R) * A * B * Mm * fmn **(-(k+1)/(2*(k-1))) - ma

    return(func1,func2)

def get_Mu(model,sta):

    Mu = model['Mm'][sta] * np.tan(model['alpha'][sta])

    return(Mu)
def get_Mr(model,sta):

    Mr = model['Mm'][sta] * np.sin(model['phi'][sta])

    return(Mr)

def get_Vr(model,sta):

    Vr = model['Vm'][sta] * np.sin(model['phi'][sta])

    return(Vr)

def get_Mx(model,sta):

    Mx = model['Mm'][sta] * np.cos(model['phi'][sta])

    return(Mx)

def get_Vx(model,sta):

    Vx = model['Vm'][sta] * np.cos(model['phi'][sta])

    return(Vx)

def get_Vm(model,sta):

    if( model['options']['gasmodel'].lower() == 'ideal'):

        Vm = model['Mm'][sta] * np.sqrt(model['k'][sta]*model['T'][sta]*model['R'][sta])

    elif( model['options']['gasmodel'].lower() == 'real'):
        print('real ga smodel -placeholder')

    else:
        print('Unable to determine gas model')
        sys.exit(1)

    return(Vm)

def check_ma(model,sta):

    ma_calc = model['Po'][sta]/np.sqrt(model['To'][sta]) \
        * np.sqrt(model['k'][sta]/model['R'][sta]) * model['A'][sta] * model['B'][sta] \
        * model['Mm'][sta] * (1 + (model['k'][sta]-1)/2 * model['M'][sta]**2.) \
        **(-(model['k'][sta]+1)/(2*(model['k'][sta]-1)))

    if( np.abs(ma_calc - model['ma'][sta]/gc) < tol ):

        print("Mass flow rate calculation does not agree")
        sys.exit(1)

    return()


def get_V(model,sta):

    V = np.sqrt(model['Vu'][sta]**2. + model['Vm'][sta]**2.)

    return(V)
def get_Vu(model,sta):

    if( sta == 0 ):
        Vu = model['Vm'][sta] * np.tan(model['alpha'][sta])

    elif( model['stations'][sta]['type'] == 'cam' ):
        Vu = model['Angmom'][sta-1] /(model['ma'][sta]/gc * model['rrms'][sta])

    elif( model['stations'][sta]['type'] == 'statorte' ):
        Vu = model['Mu'][sta] \
             * np.sqrt(model['k'][sta] * model['R'][sta] * model['T'][sta])

    elif( model['stations'][sta]['type'] == 'rotorte' ):

        Vu = (model['cp'][sta] * model['To'][sta] - model['rothalpy'][sta]) \
             /model['Urms'][sta]

    return(Vu)

def additional_calculations(model):
    '''
    calculate total pressure ratios, total temperature ratios,
    axial force on fluid, adiabatic total-to-total efficiency,
    torques due to static components, torques due to rotating components,
    polytropic total-to-total efficiencies, etc.

    These quantities are calculated relative to the previous station and
    cumulatively through the machine.
    '''
    PR = np.ones_like(model['xinner']) # total pressure ratio
    TR = np.ones_like(model['xinner']) # total temperature ratio
    TRs = np.ones_like(model['xinner']) # isentropic total temperature ratio
    Tos = np.zeros_like(model['xinner']) # isentropic total temperature
    Fx = np.zeros_like(model['xinner']) # axial force on the fluid
    dho = np.zeros_like(model['xinner']) # delta total enthalpy
    dhos = np.zeros_like(model['xinner']) # isentropic delta total enthalpy
    Tos = np.zeros_like(model['xinner']) # isentropic total temperature
    Es = np.zeros_like(model['xinner']) # isentropic power
    Es2 = np.zeros_like(model['xinner'])
    E   = np.zeros_like(model['xinner']) # actual power
    etas = np.ones_like(model['xinner'])
    etap = np.ones_like(model['xinner'])
    etasc = np.ones_like(model['xinner'])
    etapc = np.ones_like(model['xinner'])
    Tos2 = np.zeros_like(model['xinner'])

    # calculate the AXIAL FORCE on the FLUID due to the MACHINE
    # this is the NEGATIVE of the AXIAL FORCE on the MACHINE due to the FLUID
    # this is an expression of Newton's 3rd Law
    Fx[1:] =   model['ma'][1:]/gc * model['Vx'][1:]\
             - model['ma'][:-1]/gc * model['Vx'][:-1]\
             + model['P'][1:] * model['A'][1:]\
             - model['P'][:-1] *model['A'][:-1]

    PR[1:] = model['Po'][1:]/model['Po'][:-1]
    TRs    = PR**((model['k']-1)/model['k'])
    TR[1:] = model['To'][1:]/model['To'][:-1]

    for sta in range(len(model['xinner'])):

        if( sta == 0):
            Tos[sta] = TRs[sta] * model['To'][sta]
            Tos2[sta] = Tos[sta]

        else:
            Tos[sta] = TRs[sta] * model['To'][sta-1]
            Tos2[sta] = TRs[sta] * Tos[sta-1]

    dho[1:] = model['cp'][1:] * model['To'][1:]\
              - model['cp'][:-1]*model['To'][:-1]

    dhos[1:] = model['cp'][1:] * Tos[1:]\
              - model['cp'][:-1] * Tos[:-1]

    tempflag = False

    for sta in range(len(model['xinner'])):

        if( model['mbld'][sta] >= 0):
            E[sta] = model['ma'][sta]/gc * model['cp'][sta] * model['To'][sta]\
                  - model['Q'][sta]\
                  - model['mbld'][sta] * model['ma'][sta-1]/gc\
                      * model['cp'][sta] * model['Toinj'][sta]

            Es[sta] = model['ma'][sta]/gc * model['cp'][sta] * Tos[sta]\
                  - model['Q'][sta]\
                  - model['mbld'][sta] * model['ma'][sta-1]/gc\
                      * model['cp'][sta] * model['Toinj'][sta]

            Es2[sta] = Es[sta]

        else:
            E[sta] = model['ma'][sta]/gc * model['cp'][sta] * model['To'][sta]\
                  - model['Q'][sta]\
                  + model['mbld'][sta] * model['ma'][sta-1]/gc\
                      * model['cp'][sta] * model['To'][sta]

            Es[sta] = model['ma'][sta]/gc * model['cp'][sta] * Tos[sta]\
                  - model['Q'][sta]\
                  + model['mbld'][sta] * model['ma'][sta-1]/gc\
                      * model['cp'][sta] * Tos[sta]

            Es2[sta] = model['ma'][sta]/gc * model['cp'][sta] * Tos2[sta]\
                  - model['Q'][sta]\
                  + model['mbld'][sta] * model['ma'][sta-1]/gc\
                      * model['cp'][sta] * Tos[sta]


        if( not TR[sta] == 1):
            etas[sta] = (Es[sta] - E[sta-1])/(E[sta] - E[sta-1])
            etap[sta] = (model['k'][sta]-1)/model['k'][sta] \
                        * np.log(PR[sta])/np.log(TR[sta])
            tempflag = True

        if( tempflag ):
            etasc[sta] = (Es2[sta]-E[0])/(E[sta]-E[0])

            etapc[sta] = (model['k'][sta]-1)/model['k'][sta]\
                         * np.log(np.cumprod(PR)[sta])\
                         / np.log(np.cumprod(TR)[sta])

        else:
            etasc[sta] = etasc[sta-1]
            etapc[sta] = etapc[sta-1]

    # now add these quantities to the model dict
    model['Fx'] = Fx
    model['PR'] = PR
    model['TR'] = TR
    model['PRc'] = np.cumprod(PR)
    model['TRc'] = np.cumprod(TR)
    model['Es'] = Es
    model['Es2'] = Es2
    model['E'] = E
    model['etas'] = etas
    model['etap'] = etap
    model['etasc'] = etasc
    model['etapc'] = etapc
    model['Fx-static'] = Fx[model['omega'] == 0]
    model['Fx-rotate'] = Fx[model['omega'] != 0]
    model['Torque-static'] = model['Torque'][model['omega'] == 0]
    model['Torque-rotate'] = model['Torque'][model['omega'] != 0]

    return(model)
#        
#eta = np.zeros_like(xinner)
## these calculations need to be modified to account for energy leaving out the bleed
## it won't change the form much (i suspect itll only add an extra term) -ao 2.6.16
#eta = (dho_ideal - Q)/(dho - Q)
#eta[np.logical_or(np.isnan(eta),np.isinf(eta))] = 0
#
#Power = np.zeros_like(xinner)
#Power_ideal = np.zeros_like(xinner)
#Power_ideal = np.cumsum(ma*(1-mbld)*dho_ideal - Q)
#Power       = np.cumsum(ma*(1-mbld)*dho - Q)
#
#eta_overall = Power_ideal[-1]/Power[-1]
#
#
## In[72]:
#
#outfile = fn1+'.out'
#
#with open(outfile,'w') as out:
# 
#    out.write(" COMPRESSOR MEANLINE ".center(80,'#')+'\n')
#    out.write(" v0.2 1/22/2016 ".center(80,' ')+'\n\n')
#    out.write("{0:<10s} {1:<10s} {2:<10s} {3:<10s} {4:<10s} {5:<10s}\n".format( 'Nc[RPM]','mc[lbm/s]',\
#		    'Po[psfa]','To[deg R]','alpha[deg]','phi[deg]'))
#
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.3f} {3:<10.3f} {4:<10.3f} {5:<10.3f}\n".format(\
#		    Nc_inlet,mc[0],Po[0], To[0],alpha_inlet*rad2deg, phi[0]*rad2deg))
#    out.write("\n{0:<10s} {1:<10s}\n".format('gasmodel','relhumidity'))
#    out.write("{0:<10s} {1:<10.3f}\n\n".format(gasmodel,relhumidity))
#    
#    for sta in range(len(xinner)):
#        if( sta == 0):
#            #out.write("#"*80+'\n')
#            out.write(" STATION INFORMATION ".center(80,'#')+"\n")
#            #out.write("#"*80+'\n\n')
#        else:
#            out.write("#"*80+"\n")
#    
#    out.write("{0:<10s} {1:<10s} {2:<10s}\n".format('Station','Label','Type'))
#    out.write("{0:<10d} {1:<10s} {2:<10s}\n\n".format(sta,label[sta],statype[sta]))
#    out.write("{0:<10s} {1:<10s} {2:<10s} {3:<10s} {4:<10s} {5:<10s} {6:<10s}\n".format(\
#		    'xinner[in]','xouter[in]','rinner[in]','router[in]','alpha[deg]',\
#		    'phi[deg]','Area[ft**2]'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.3f} {3:<10.3f} {4:<10.3f} {5:<10.3f} {6:<10.3f}\n\n".format(\
#		    xinner[sta]*ft2in,xouter[sta]*ft2in,rinner[sta]*ft2in,router[sta]*ft2in,\
#		    alpha[sta]*rad2deg,phi[sta]*rad2deg,A[sta]))
#    
#    out.write("{0:<10s} {1:<10s} {2:<10s} {3:<10s} {4:<10s} {5:<10s} {6:<10s}\n".format(\
#		    'xrms[in]','rrms[in]','ma[lbm/s]','mbld[-]','Q[HP]','cp/cv[-]','R[lbf-ft/slg-R]'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.3f} {3:<10.3f} {4:<10.3f} {5:<10.3f} {6:<10.3f}\n\n".format(\
#		    xrms[sta]*ft2in,rrms[sta]*ft2in,ma[sta],mbld[sta],Q[sta]*lbfft_s2HP,k[sta],R[sta]))
#    
#    if( statype[sta].lower() == 'statorte'):
#        out.write("{0:<10s} {1:<10s} {2:<15s} {3:<10s}\n".format( 'Po[psfa]','To[deg R]','po[slg/cf]','Ptloss[-]'))
#        out.write("{0:<10.3f} {1:<10.3f} {2:<10.6f} {3:<10.3f}\n\n".format( Po[sta],To[sta],po[sta],Ptloss[sta]))
#    
#    elif( statype[sta].lower() == 'rotorte'):
#        out.write("{0:<10s} {1:<10s} {2:<10s} {3:<10s} {4:<10s}\n".format( 'Po[psfa]','To[deg R]','po[slg/cf]','Work CF[-]','Flow CF[-]'))
#        out.write("{0:<10.3f} {1:<10.3f} {2:<10.6f} {3:<10.3f} {4:<10.3f}\n\n".format( Po[sta],To[sta],po[sta],workcf[sta],flowcf[sta]))
#    
#    else:
#        out.write("{0:<10s} {1:<10s} {2:<10s}\n".format( 'Po[psfa]','To[deg R]','po[slg/cf]'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.6f}\n\n".format( Po[sta],To[sta],po[sta]))
#
#    out.write("{0:<10s} {1:<10s} {2:<10s}\n".format( 'P[psfa]','T[deg R]','p[slg/cf]'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.6f}\n\n".format( P[sta],T[sta],p[sta]))
#    
#    out.write("{0:<10s} {1:<10s} {2:<10s} {3:<10s} {4:<10s} {5:<10s}{6:<10s}\n".format( 'Vx[ft/s]','Vr[ft/s]','Vu[ft/s]','Vm[ft/s]','V[ft/s]','Vurel[ft/s]','Vrel[ft/s]'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.3f} {3:<10.3f} {4:<10.3f} {5:<10.3f} {6:<10.3f}\n\n".format( Vx[sta],Vr[sta],Vu[sta],Vm[sta],V[sta],Vurel[sta],Vrel[sta]))
#    
#    out.write("{0:<10s} {1:<10s} {2:<10s} {3:<10s} {4:<10s} {5:<10s} {6:<10s}\n".format( 'Mx[-]','Mr[-]','Mu[-]','Mm[-]','M[-]','Murel[-]','Mrel[-]'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.3f} {3:<10.3f} {4:<10.3f} {5:<10.3f} {6:<10.3f}\n\n".format( Mx[sta],Mr[sta],Mu[sta],Mm[sta],M[sta],Murel[sta],Mrel[sta]))
#    
#    out.write("{0:<10s} {1:<10s}\n".format('T[lbf-ft]','mc[lbm/s]'))
#    out.write("{0:<10.3f} {1:<10.3f}\n\n".format(Torque[sta],mc[sta]))
#    
#    out.write(' OVERALL '.center(80,'#')+'\n')
#    
#    out.write("{0:<10s} {1:<10s} {2:<10s} {3:<10s}\n".format( 'TR[-]','PR[-]','Ad. Eff[-]','Power [HP]'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.3f} {3:<10.3f}\n\n".format( np.cumprod(TR)[-1],np.cumprod(PR)[-1],eta_overall,Power[-1]*lbfft_s2HP))
#    
#    out.write("{0:<10s} {1:<10s} {2:<10s}\n".format('Fx[lbf]','Fx(static)','Fx(rotate)'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.3f}\n\n".format(np.cumsum(Fx)[-1],
#    np.cumsum(Fx_static)[-1],np.cumsum(Fx_rotate)[-1]))
#    
#    out.write("{0:<10s} {1:<10s} {2:<10s}\n".format('T[lbf-ft]','T(static)','T(rotate)'))
#    out.write("{0:<10.3f} {1:<10.3f} {2:<10.3f}".format( np.cumsum(Torque)[-1],np.cumsum(Torque_static)[-1],np.cumsum(Torque_rotate)[-1]))
# 
#
#
## plotting
#plt.figure(0,figsize=(5*np.e,5))
#plt.plot(xinner * ft2in,rinner * ft2in,'bo-',label=r'Endwall')
#plt.plot(xouter * ft2in,router * ft2in,'bo-')
#
#for i in range(len(stator_xle)):
#    if( i % 2 == 0):
#        plt.plot(stator_xle[i:i+2] * ft2in,stator_rle[i:i+2] * ft2in,'g--')
#        plt.plot(stator_xte[i:i+2] * ft2in,stator_rte[i:i+2] * ft2in,'g--')
#    
#for i in range(len(rotor_xle)):
#    if( i % 2 == 0):
#        plt.plot(rotor_xle[i:i+2] * ft2in,rotor_rle[i:i+2] * ft2in,'r--')
#        plt.plot(rotor_xte[i:i+2] * ft2in,rotor_rte[i:i+2] * ft2in,'r--')
#    
#plt.plot(xinlet * ft2in,rinlet * ft2in,color='b',linestyle='dotted')
#plt.plot(xexit * ft2in,rexit * ft2in,color='b',linestyle='dotted')
#plt.plot(xrms * ft2in,rrms * ft2in,'mx-.',label=r'RMS')
#plt.axis('equal')
##plt.grid('on')
##plt.legend(loc='best')
##plt.ylim(ymin=0)
#plt.ylabel(r'Radius; $r\ [in]$',fontsize='x-large')
#plt.xlabel(r'Axial; $x\ [in]$',fontsize='x-large')
#plt.grid('on')
#plt.savefig('./geometry.pdf')
#
#plt.figure(1,figsize=(5*np.e,5))
#plt.plot(xrms*ft2in,Mrel,'ro-',label=r'$M_{rel}$')
#plt.plot(xrms*ft2in,M,'ko-',label=r'$M$')
#plt.plot(xrms*ft2in,Mm,'bo-',label=r'$M_m$')
#plt.plot(xrms*ft2in,abs(Murel),'go-',label=r'$M_{u,rel}$')
#plt.plot(xrms*ft2in,Mu,'mo-',label=r'$M_u$')
#plt.legend(loc='best')
#plt.grid('on')
#plt.xlabel(r'Axial; $x\ [in]$',fontsize='x-large')
#plt.ylim(ymax=1.)
#plt.savefig('./M.pdf')
#
#plt.figure(2,figsize=(5*np.e,5))
#plt.plot(xrms*ft2in,A,'bo-')
#plt.grid('on')
#plt.ylabel(r'Perpendicular Flow Area; $A_\perp\ [ft^2]$',fontsize='x-large')
#plt.xlabel(r'Axial; $x\ [in]$',fontsize='x-large')
#
#plt.figure(3,figsize=(5*np.e,5))
#plt.plot(xrms*ft2in,Po,'bo-',label=r'$P_o$')
#plt.plot(xrms*ft2in,P,'bo--',label=r'$P$')
#plt.legend(loc='best')
#plt.grid('on')
#plt.ylabel(r'Pressure $[lb_f/ft^2]$',fontsize='x-large')
#plt.xlabel(r'Axial; $x\ [in]$',fontsize='x-large')
#
#plt.figure(4,figsize=(5*np.e,5))
#plt.plot(xrms*ft2in,To,'ro-',label=r'$T_o$')
#plt.plot(xrms*ft2in,T,'ro--',label=r'$T$')
#plt.legend(loc='best')
#plt.grid('on')
#plt.ylabel(r'Temperature $[deg\ R]$',fontsize='x-large')
#plt.xlabel(r'Axial; $x\ [in]$',fontsize='x-large')
#
#

def create_help(model):
    model['help'] = {}
    model['help']['alpha']  = 'Air angle defined from axial in abs. frame: tan(alpha) = Vu/Vm'
    model['help']['beta']   = 'Air angle defined from axial in rel. frame: tan(beta) = (u-Vu)/Vm'
    model['help']['gear']   = 'Gear Ratio: ratio of the rotation rate of the calculation station relative to the inlet Nm value'

    model['help']['PR']     = 'Rotor Total Pressure Ratio'
    model['help']['rrms']   = 'root mean square radius of the calculation station'
    model['help']['workcf'] = 'Rotor Work Coefficient at rrms: dho/u**2'
    model['help']['xrms']   = 'axial location of rrms of the calculation station'
    model['help']['solidity'] =  'ratio of airfoil chord to circumferential pitch, uses rrms'
    return(model)

def create_units(model):
    model['units'] = {}
    model['units']['P'] = 'absolute pounds force per square foot [psfa] or [lbf/ft**2]'
    model['units']['T'] = 'degree Rankine [deg R]'
    model['units']['p'] = 'slugs per cubic foot [slg/ft**3]'
    model['units']['R'] = 'pound force feet per slug per degree Rankine [lbfft/(slg-R)]'
    model['units']['m'] = 'pound mass per second [lbm/s]'
    model['units']['V'] = 'feet per second [ft/s]'
    model['units']['U'] = 'feet per second [ft/s]'
    model['units']['gc']= 'feet per second squared [ft/s**2]'
    return(model)

def output(args,model):

    fn = args.fn.split('.')[0]

    # write output yaml file
    if( model['options']['yamlout'] ):
        with open(fn+'.yaml','w') as out:
            out.write(yaml.dump(model))

    # write output file in different format
#    if( model['options']['asciiout'] ):
#        with open(fn+'.out','w') as out:

    # create pdf plots
#    if( model['options']['plots'] ):
    return()

if(__name__ == '__main__'):
    main()
