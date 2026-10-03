import csv, sys
tag = sys.argv[1] if len(sys.argv) > 1 else 'repaired'
for name in sys.argv[2:]:
    rows = list(csv.DictReader(open(f'benchmarks/vehicle_dynamics/{tag}/{name}.csv')))
    print(f'== {name} ==')
    for r in rows[-8:]:
        print(f"  t={r['t']} vx={float(r['vx']):5.1f} vy={float(r['vy']):+6.2f} wz={float(r['yaw_rate']):+6.2f} "
              f"beta={float(r['sideslip']):+5.2f} a_f={float(r['alpha_f']):+5.2f} a_r={float(r['alpha_r']):+5.2f} "
              f"fyf={float(r['fy_f']):+6.0f} fyr={float(r['fy_r']):+6.0f}")
