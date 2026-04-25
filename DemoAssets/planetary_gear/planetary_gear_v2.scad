// planetary_gear_v2.scad
// Identical assembly geometry to planetary_gear.scad, but sun and planet teeth
// use modul_v2=2.5 instead of modul=3.  This creates ≥2.4 mm clearance at every
// tooth-pair interface, making the set stable under d_hat=1e-3 m IPC simulation.
//
// Clearance analysis (modul_v2=2.88, centre-distance=31.5 mm, ring at modul=3):
//   Sun–planet normal backlash : 2.37 mm  (flank gap ≈ 1.18 mm → gears engage quickly)
//   Ring inner wall → planet tip : 1.41 mm > d_hat=1 mm ✓
//   Ring tooth tip  → planet root: 1.14 mm > d_hat=1 mm ✓  ← binding constraint
//
// Export:
//   openscad -o sun_gear_handle_v2.stl -D "part=5" planetary_gear_v2.scad
//   openscad -o planet_gear_v2.stl     -D "part=2" planetary_gear_v2.scad
//   ring_gear.stl / carrier.stl are UNCHANGED from v1.
//
// part: 0=assembly  1=sun  2=planet  3=ring  4=carrier  5=sun+handle

use <gears.scad>

part = 0;

// ── Shared geometry (centre distances, assembly layout) ──────────────────────
modul          = 3;       // kept for centre-distance + ring/carrier geometry
modul_v2       = 2.88;    // ~1 mm tooth gap at all interfaces for d_hat=1e-3

sun_teeth      = 12;
planet_teeth   = 9;
num_planets    = 3;
gear_width     = 12;      // mm
rim_width      = 12;      // mm
bore           = 8.5;     // mm  (8 mm pin + 0.5 mm clearance)
pressure_angle = 20;
helix_angle    = 0;

carrier_thick  = 4.5;     // mm
pin_r          = 4.0;     // pin radius (mm)
pin_h          = 21;      // pin height (mm)
chamfer        = 1.0;     // bore chamfer (mm)

// Carrier flange
flange_thick     = 6;
flange_r         = 53;
flange_inner_r   = 6;
flange_bolt_r    = 49;
flange_bolt_n    = 6;
flange_bolt_hole = 3.5;

// Handle (sun crank)
handle_post_r       = bore/2;
handle_insert_depth = 3.0;    // match v1 — leaves bore open Z=[0,9] for support_pin
handle_arm_r        = 2.0;
handle_fillet_r     = 4.0;
handle_post_h       = 8.0;
handle_arm_len      = 60.0;
handle_grip_h       = 18.0;
handle_ball_r       = 4.0;
handle_angle        = 60.0;

// ── Derived ──────────────────────────────────────────────────────────────────
ring_teeth       = sun_teeth + 2 * planet_teeth;            // 30
center_distance  = modul * (sun_teeth + planet_teeth) / 2;  // 31.5 mm
need_rotate_sun  = (planet_teeth % 2 == 0) ? 1 : 0;

// ── Arc fillet helper ────────────────────────────────────────────────────────
module arc_fillet(R_arc, r_tube, a_start, a_end, N=16) {
    union() {
        for (i = [0 : N-1]) {
            a0 = a_start + i   * (a_end - a_start) / N;
            a1 = a_start + (i+1) * (a_end - a_start) / N;
            hull() {
                translate([R_arc * cos(a0), 0, R_arc * sin(a0)])
                    sphere(r=r_tube, $fn=12);
                translate([R_arc * cos(a1), 0, R_arc * sin(a1)])
                    sphere(r=r_tube, $fn=12);
            }
        }
    }
}

// ── Sun gear (v2 tooth profile) ───────────────────────────────────────────────
module make_sun() {
    color("gold")
    rotate([0, 0, 180/sun_teeth * need_rotate_sun])
        spur_gear(modul_v2, sun_teeth, gear_width, bore,
                  pressure_angle, helix_angle, optimized=true);
}

// ── Sun gear with handle (v2 tooth profile) ──────────────────────────────────
module make_sun_with_handle() {
    color("gold") union() {
        rotate([0, 0, 180/sun_teeth * need_rotate_sun])
            spur_gear(modul_v2, sun_teeth, gear_width, bore,
                      pressure_angle, helix_angle, optimized=true);

        rotate([0, 0, handle_angle]) {
            _Z0 = gear_width + handle_post_h;

            // Post inside bore
            translate([0, 0, gear_width - handle_insert_depth])
                cylinder(r=handle_post_r, h=handle_insert_depth, $fn=48);

            // Taper above gear top
            hull() {
                translate([0, 0, gear_width])
                    cylinder(r=handle_post_r, h=0.01, $fn=48);
                translate([0, 0, _Z0 - handle_fillet_r - 0.01])
                    cylinder(r=handle_arm_r,  h=0.01, $fn=48);
            }

            // Corner-1 arc: post → arm
            translate([handle_fillet_r, 0, _Z0 - handle_fillet_r])
                arc_fillet(handle_fillet_r, handle_arm_r, 180, 90);

            // Horizontal arm
            translate([handle_fillet_r, 0, _Z0])
                rotate([0, 90, 0])
                    cylinder(r=handle_arm_r,
                             h=handle_arm_len - 2*handle_fillet_r, $fn=32);

            // Corner-2 arc: arm → grip
            translate([handle_arm_len - handle_fillet_r, 0, _Z0 + handle_fillet_r])
                arc_fillet(handle_fillet_r, handle_arm_r, 270, 360);

            // Vertical grip
            translate([handle_arm_len, 0, _Z0 + handle_fillet_r])
                cylinder(r=handle_arm_r,
                         h=handle_grip_h - handle_fillet_r, $fn=32);

            // Grip ball
            translate([handle_arm_len, 0, _Z0 + handle_grip_h])
                sphere(r=handle_ball_r, $fn=32);
        }
    }
}

// ── Planet gear (v2 tooth profile) ──────────────────────────────────────────
module make_planet() {
    color("steelblue")
    difference() {
        spur_gear(modul_v2, planet_teeth, gear_width, bore,
                  pressure_angle, helix_angle, optimized=true);
        // Bottom chamfer
        translate([0, 0, -0.01])
            cylinder(h=chamfer+0.01, r1=bore/2+chamfer, r2=bore/2, $fn=48);
        // Top chamfer
        translate([0, 0, gear_width-chamfer])
            cylinder(h=chamfer+0.01, r1=bore/2, r2=bore/2+chamfer, $fn=48);
    }
}

// ── Ring gear (v1, modul=3 – UNCHANGED) ─────────────────────────────────────
module make_ring() {
    color("firebrick")
    ring_gear(modul, ring_teeth, gear_width, rim_width,
              pressure_angle, helix_angle);
}

// ── Carrier (UNCHANGED) ──────────────────────────────────────────────────────
module make_carrier() {
    plate_r = center_distance + modul * planet_teeth / 2 + 2;
    color("silver") {
        difference() {
            cylinder(r=plate_r, h=carrier_thick, $fn=120);
            translate([0, 0, -0.5])
                cylinder(r=bore/2, h=carrier_thick+1, $fn=48);
        }
        for (n = [0 : num_planets-1]) {
            a = n * 360 / num_planets;
            translate([center_distance * cos(a),
                       center_distance * sin(a), 0])
                cylinder(r=pin_r, h=pin_h, $fn=48);
        }
        // Output flange on bottom face
        translate([0, 0, -flange_thick])
            difference() {
                cylinder(r=flange_r, h=flange_thick, $fn=120);
                translate([0, 0, -0.5])
                    cylinder(r=flange_inner_r, h=flange_thick+1, $fn=48);
                for (i = [0 : flange_bolt_n-1])
                    rotate([0, 0, i * 360 / flange_bolt_n])
                        translate([flange_bolt_r, 0, -0.5])
                            cylinder(r=flange_bolt_hole, h=flange_thick+1, $fn=24);
            }
    }
}

// ── Assembly ─────────────────────────────────────────────────────────────────
module assembly() {
    make_sun_with_handle();

    for (n = [0 : num_planets-1]) {
        a = n * 360 / num_planets;
        translate([center_distance * cos(a),
                   center_distance * sin(a), 0])
            rotate([0, 0, n * 360 * (modul*sun_teeth) / (modul_v2*planet_teeth)])
                make_planet();
    }

    make_ring();

    translate([0, 0, -gear_width/2])
        make_carrier();
}

// ── Select part ──────────────────────────────────────────────────────────────
if      (part == 1) make_sun();
else if (part == 5) make_sun_with_handle();
else if (part == 2) make_planet();
else if (part == 3) make_ring();
else if (part == 4) make_carrier();
else                assembly();
