import h5py

# Open and explore the file structure
with h5py.File(r'C:\Users\Noel\Desktop\Biotech\DECONOMIX_DATA\GSM6885374.h5', 'r') as f:
    print("=" * 50)
    print("RAW HDF5 FILE STRUCTURE")
    print("=" * 50)
    
    def print_structure(name, obj):
        print(name)
    
    f.visititems(print_structure)
    
    print("\n" + "=" * 50)
    print("TOP-LEVEL GROUPS")
    print("=" * 50)
    for key in f.keys():
        print(f"- {key}")
        if isinstance(f[key], h5py.Group):
            print(f"  Contains: {list(f[key].keys())}")