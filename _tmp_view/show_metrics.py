import json, sys
r = json.load(open('benchmarks/vehicle_dynamics/repaired/results.json'))
keys = sys.argv[1:] if len(sys.argv) > 1 else list(r)
for k in keys:
    m = r[k]
    print(f"{k:42s} peak={m['peak_sideslip_deg']:6.1f} steady={m['steady_sideslip_deg']:7.1f} final={m['final_sideslip_deg']:7.1f} "
          f"wz_ss={m['steady_yaw_rate_dps']:7.1f} vy_f={m['final_vy_ms']:6.2f} spd_f={m['final_speed_ms']:5.1f}")
