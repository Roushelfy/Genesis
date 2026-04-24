// Planetary Gear Set using Dr. Joerg Janssen's gear library.
// Export:  openscad.com -o out.stl -D "part=N" planetary_gear.scad
//   part: 0=assembly  1=sun  2=planet  3=ring  4=carrier

use <gears.scad>

part = 0;

modul          = 3;       // 3 mm  (was 2; ×1.5 scale)
sun_teeth      = 12;
planet_teeth   = 9;
num_planets    = 3;
gear_width     = 12;      // 12 mm (was 8; ×1.5)
rim_width      = 12;      // 12 mm (was 8; ×1.5)
bore           = 8.5;     // 8.5 mm diameter (pin 8mm + 0.5mm clearance)
pressure_angle = 20;
helix_angle    = 0;

carrier_thick  = 4.5;     // 4.5 mm (was 3; ×1.5)
pin_r          = 4.0;     // 4.0 mm radius (8 mm diameter), bore=10 so 1mm clearance per side
pin_h          = 21;      // 21 mm (was 14; ×1.5)
chamfer        = 1.0;     // bore chamfer height & depth (mm)

// Mounting flange (carrier bottom face, for robot-arm attachment to the output)
// Sits below carrier plate; outer radius slightly larger than carrier plate (r≈47mm)
flange_thick     = 6;     // mm – flange plate thickness
flange_r         = 53;    // mm – flange outer radius
flange_inner_r   = 6;     // mm – central clearance hole (≥ bore/2=4.25mm)
flange_bolt_r    = 49;    // mm – bolt hole circle radius
flange_bolt_n    = 6;     // number of bolt holes
flange_bolt_hole = 3.5;   // mm – bolt hole radius

// Support pin – a fixed cylindrical shaft on the gear axis.
// Provides a rotation axis for sun gear + carrier and prevents the carrier
// from falling (shoulder wider than carrier bore but narrower than flange hole).
//
//  part=6 exports the pin in its own local frame (bottom at z=0).
//  In assembly it is translated to z = -(gear_width/2 + flange_thick).
//
support_pin_r          = bore/2 - 0.25;          // 4.0 mm shaft radius (0.25 mm clearance each side)
support_pin_shoulder_r = bore/2 + 1.0;           // 5.25 mm – 1.0 mm radial gap from bore inner
                                                  //  wall; 0.75 mm gap from flange inner hole (6 mm)
support_pin_shoulder_h = 1.5;                     // mm – shoulder thickness
// Shoulder top is placed 1 mm BELOW the carrier-plate bottom face (assembly z=−6 mm)
// to avoid the sharp inner step at the bore/flange transition.
//   shoulder top  in pin-local coords = gear_width/2 − 1 = 5 mm  → assembly z = −7 mm
//   shoulder base in pin-local coords = 5 − 1.5 = 3.5 mm        → assembly z = −8.5 mm
// Total height: from flange-bottom (assembly z=−12) to 1mm below handle-insert bottom.
// handle_insert_depth=3 mm → handle insert occupies z=9→12 mm → pin top at z=8 mm.
//   pin_h = (gear_width - handle_insert_depth - 1) + gear_width/2 + flange_thick
//         = (12 - 3 - 1) + 6 + 6 = 20 mm
support_pin_h = 20;   // mm

// Handle – L-shaped crank: post inserts into bore, tapers above gear, then horizontal arm + grip
handle_post_r       = bore/2;    // post radius = bore inner radius (exact fit into bore)
handle_insert_depth = 3.0;       // depth the post inserts into the bore (mm)
handle_arm_r        = 2.0;       // arm & grip rod radius (mm)
handle_fillet_r     = 4.0;       // arc fillet radius at each L-bend (mm)
handle_post_h       = 8.0;       // post height above gear top face before fillet (mm)
handle_arm_len      = 60.0;      // horizontal arm length from centre to grip axis (mm)
handle_grip_h       = 18.0;      // vertical grip height above horizontal arm (mm)
handle_ball_r       = 4.0;       // grip ball radius (mm)
handle_angle        = 60.0;      // deg – direction of arm (60° clears the bottom planet gear)

// Derived
ring_teeth       = sun_teeth + 2 * planet_teeth;
center_distance  = modul * (sun_teeth + planet_teeth) / 2;
d_sun            = modul * sun_teeth;
d_planet         = modul * planet_teeth;
need_rotate_sun  = (planet_teeth % 2 == 0) ? 1 : 0;

// ── Parts ──

module make_support_pin() {
    // Pin in local frame: bottom at z = 0.
    // In assembly: translate([0, 0, -(gear_width/2 + flange_thick)]).
    //
    // Geometry:
    //   Shaft:    r = support_pin_r    (fits in bore + flange inner hole)
    //   Shoulder: r = support_pin_shoulder_r,  placed so its top face aligns with
    //             the carrier-plate bottom (local z = gear_width/2).
    //             Blocks the carrier from sliding downward.
    color("dimgray")
    union() {
        // Main shaft through the entire stack
        cylinder(r=support_pin_r, h=support_pin_h, $fn=48);

        // Support shoulder – local z: (gear_width/2 − 1 − shoulder_h) → (gear_width/2 − 1)
        // Assembly z: −8.5 mm → −7 mm  (1 mm below carrier-plate bottom at −6 mm)
        // Shoulder stays inside flange inner hole (r=5.25 < 6), sits below bore/flange step.
        translate([0, 0, gear_width/2 - 1 - support_pin_shoulder_h])
            cylinder(r=support_pin_shoulder_r, h=support_pin_shoulder_h, $fn=48);
    }
}

module make_sun() {
    color("gold")
    rotate([0, 0, 180/sun_teeth * need_rotate_sun])
        spur_gear(modul, sun_teeth, gear_width, bore,
                  pressure_angle, helix_angle, optimized=true);
}

// Quarter-arc fillet in the XZ plane.
// Sweeps a tube of radius r_tube along an arc of radius R_arc,
// from a_start to a_end degrees (measured from +X axis in the XZ plane).
// Hull-of-sphere pairs produces a smooth manifold solid.
module arc_fillet(R_arc, r_tube, a_start, a_end, N=16) {
    union() {
        for (i = [0 : N-1]) {
            a0 = a_start + i  * (a_end - a_start) / N;
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

module make_sun_with_handle() {
    color("gold") union() {
        // Gear body
        rotate([0, 0, 180/sun_teeth * need_rotate_sun])
            spur_gear(modul, sun_teeth, gear_width, bore,
                      pressure_angle, helix_angle, optimized=true);

        rotate([0, 0, handle_angle]) {
            _Z0 = gear_width + handle_post_h;   // arm centre-line height

            // 1a. Post inside bore: same diameter as bore, inserts handle_insert_depth mm down
            translate([0, 0, gear_width - handle_insert_depth])
                cylinder(r=handle_post_r, h=handle_insert_depth, $fn=48);

            // 1b. Taper above gear top face: bore diameter → arm diameter
            //     (gives a smooth visual transition from thick post to thin arm)
            hull() {
                translate([0, 0, gear_width])
                    cylinder(r=handle_post_r, h=0.01, $fn=48);
                translate([0, 0, _Z0 - handle_fillet_r - 0.01])
                    cylinder(r=handle_arm_r,  h=0.01, $fn=48);
            }

            // 2. Corner-1 arc: post (+Z) → arm (+X)
            //    Arc centre at (handle_fillet_r, 0, _Z0 - handle_fillet_r)
            //    Sweeps CW from 180° (post end) to 90° (arm start)
            translate([handle_fillet_r, 0, _Z0 - handle_fillet_r])
                arc_fillet(handle_fillet_r, handle_arm_r, 180, 90);

            // 3. Horizontal arm (straight section between the two fillet arcs)
            translate([handle_fillet_r, 0, _Z0])
                rotate([0, 90, 0])
                    cylinder(r=handle_arm_r,
                             h=handle_arm_len - 2*handle_fillet_r,
                             $fn=32);

            // 4. Corner-2 arc: arm (+X) → grip (+Z)
            //    Arc centre at (handle_arm_len - handle_fillet_r, 0, _Z0 + handle_fillet_r)
            //    Sweeps CCW from 270° (arm end) to 360° (grip start)
            translate([handle_arm_len - handle_fillet_r, 0, _Z0 + handle_fillet_r])
                arc_fillet(handle_fillet_r, handle_arm_r, 270, 360);

            // 5. Vertical grip (shortened at bottom for fillet arc)
            translate([handle_arm_len, 0, _Z0 + handle_fillet_r])
                cylinder(r=handle_arm_r,
                         h=handle_grip_h - handle_fillet_r,
                         $fn=32);

            // 6. Grip ball
            translate([handle_arm_len, 0, _Z0 + handle_grip_h])
                sphere(r=handle_ball_r, $fn=32);
        }
    }
}

module make_planet() {
    color("steelblue")
    difference() {
        spur_gear(modul, planet_teeth, gear_width, bore,
                  pressure_angle, helix_angle, optimized=true);
        // Bottom chamfer
        translate([0, 0, -0.01])
            cylinder(h=chamfer+0.01, r1=bore/2+chamfer, r2=bore/2, $fn=48);
        // Top chamfer
        translate([0, 0, gear_width-chamfer])
            cylinder(h=chamfer+0.01, r1=bore/2, r2=bore/2+chamfer, $fn=48);
    }
}

module make_ring() {
    color("firebrick")
    ring_gear(modul, ring_teeth, gear_width, rim_width,
              pressure_angle, helix_angle);
}

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
        // Output mounting flange on bottom face (z = -flange_thick … 0)
        translate([0, 0, -flange_thick])
            difference() {
                cylinder(r=flange_r, h=flange_thick, $fn=120);
                // Central clearance hole
                translate([0, 0, -0.5])
                    cylinder(r=flange_inner_r, h=flange_thick+1, $fn=48);
                // Bolt holes
                for (i = [0 : flange_bolt_n-1])
                    rotate([0, 0, i * 360 / flange_bolt_n])
                        translate([flange_bolt_r, 0, -0.5])
                            cylinder(r=flange_bolt_hole, h=flange_thick+1, $fn=24);
            }
    }
}

// ── Assembly ──

module assembly() {
    make_sun_with_handle();

    for (n = [0 : num_planets-1]) {
        a = n * 360 / num_planets;
        translate([center_distance * cos(a),
                   center_distance * sin(a), 0])
            rotate([0, 0, n * 360 * d_sun / d_planet])
                make_planet();
    }

    make_ring();

    translate([0, 0, -gear_width/2])
        make_carrier();

    // Support pin: bottom at z = -(gear_width/2 + flange_thick) = -12 mm
    translate([0, 0, -(gear_width/2 + flange_thick)])
        make_support_pin();
}

// ── Select ──
//   part=1 : sun gear (no handle)
//   part=5 : sun gear with handle
if      (part == 1) make_sun();
else if (part == 5) make_sun_with_handle();
else if (part == 2) make_planet();
else if (part == 3) make_ring();
else if (part == 4) make_carrier();
else if (part == 6) make_support_pin();
else                assembly();
