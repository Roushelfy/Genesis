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

// Derived
ring_teeth       = sun_teeth + 2 * planet_teeth;
center_distance  = modul * (sun_teeth + planet_teeth) / 2;
d_sun            = modul * sun_teeth;
d_planet         = modul * planet_teeth;
need_rotate_sun  = (planet_teeth % 2 == 0) ? 1 : 0;

// ── Parts ──

module make_sun() {
    color("gold")
    rotate([0, 0, 180/sun_teeth * need_rotate_sun])
        spur_gear(modul, sun_teeth, gear_width, bore,
                  pressure_angle, helix_angle, optimized=true);
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
    }
}

// ── Assembly ──

module assembly() {
    make_sun();

    for (n = [0 : num_planets-1]) {
        a = n * 360 / num_planets;
        translate([center_distance * cos(a),
                   center_distance * sin(a), 0])
            rotate([0, 0, n * 360 * d_sun / d_planet])
                make_planet();
    }

    make_ring();

    translate([0, 0, -gear_width/2 - carrier_thick])
        make_carrier();
}

// ── Select ──
if      (part == 1) make_sun();
else if (part == 2) make_planet();
else if (part == 3) make_ring();
else if (part == 4) make_carrier();
else                assembly();
