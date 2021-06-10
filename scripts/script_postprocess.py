import shutil
import os
from display.meta.two_d_map import make_2d_map
from display.diagrams.sweeper_plots import sweeper_plots


def full_analysis(folder_path,
                  i, i_values, j, j_values,
                  mfp_dict, bw_norm, process_guesses):

    frf = os.path.join(folder_path, 'Final_Results')
    # Process final sols
    if os.path.exists(frf):
        sweeper_plots(i, i_values, j, j_values, mfp_dict,
                      frf, BW_norm=bw_norm, show=False)
    else:
        print("Solutions not found")
    # process guesses
    if process_guesses:
        guesses_folder = os.path.join(folder_path, 'Guesses_Results')
        for guess in os.listdir(guesses_folder):
            guess = os.path.join(guesses_folder, guess)
            print('Processing guess:', guess)
            sweeper_plots(i, i_values, j, j_values, mfp_dict,
                          guess, BW_norm=bw_norm)


def postprocess(results_folder, batch_folder, folder_list,
                i, i_values, j, j_values,
                mfp_dict, bw_norm,
                full=True, process_guesses=False,
                make_map=False):

    if full:
        Diagrams_Folder = os.path.join(results_folder, batch_folder+'_diags')
        os.makedirs(Diagrams_Folder, exist_ok=True)

        for folder in folder_list:
            folder_path = os.path.join(results_folder, batch_folder, folder)
            print('Processing: ', folder)
            full_analysis(
                folder_path,
                i, i_values, j, j_values,
                mfp_dict, bw_norm,
                process_guesses=process_guesses
                )

            # Copy only diagrams to diag_folder
            diagram = os.path.join(
                folder_path, 'Final_Results', 'Plots', 'PhaseDiagram.png')
            out = os.path.join(Diagrams_Folder, folder+'.png')
            shutil.copy(diagram, out)

    if make_map:
        make_2d_map(results_folder, batch_folder, bw_norm)
