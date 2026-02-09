import numpy as np
import os
import sunkit_image
from sunkit_image import asda

# --- CRITICAL FIX: Forces saving to the visible Project Folder ---
LOCAL_PROJECT_PATH = os.path.join(os.getcwd(), "sunkit_image", "data", "test")
# -----------------------------------------------------------------

def get_local_test_filepath(filename):
    # We read the input from the installed system files
    base_dir = os.path.dirname(sunkit_image.__file__)
    return os.path.join(base_dir, "data", "test", filename)

def regenerate_asda_data():
    # 1. Locate the input data
    vel_file = get_local_test_filepath("asda_vxvy.npz")
    print(f"Reading input from: {vel_file}")
    
    # Load the data
    vxvy = np.load(vel_file, allow_pickle=True)
    vx = vxvy["vx"]
    vy = vxvy["vy"]
    data = vxvy["data"]

    # 2. Run the ASDA algorithm
    print("Running ASDA algorithm...")
    factor = 1
    r = 3
    gamma = asda.calculate_gamma_values(vx, vy, factor, r)
    center_edge = asda.get_vortex_edges(gamma)
    ve, vr, vc, ia = asda.get_vortex_properties(vx, vy, center_edge, data)

    # 3. SAVE TO THE PROJECT FOLDER
    output_path = os.path.join(LOCAL_PROJECT_PATH, "asda_correct.npz")
    print(f"Saving regenerated data to: {output_path}")
    
    # --- CATCH-ALL FIX: Save EVERY key found ---
    # This prevents 'KeyError: radius' and 'KeyError: peak'
    save_dict = {
        "ve": np.array(ve, dtype=object),
        "vr": np.array(vr, dtype=object),
        "vc": np.array(vc, dtype=object),
        "ia": np.array(ia, dtype=object),
    }
    
    # Automatically add center, edge, peak, radius, etc.
    print("Keys found:", list(center_edge.keys()))
    for key, value in center_edge.items():
        save_dict[key] = np.array(value, dtype=object)

    np.savez(output_path, **save_dict)
    print("Done! NOW run 'git status' - you should see red text.")

if __name__ == "__main__":
    regenerate_asda_data()