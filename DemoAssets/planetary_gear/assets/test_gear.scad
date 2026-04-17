pi = 3.14159265359;
function inv_angle(br, r) = (r <= br) ? 0 : sqrt(r*r - br*br)/br * 180/pi;

module ext_gear_2d(n, m, pa, bl) {
    pr  = m*n/2;
    br  = pr*cos(pa);
    ar  = pr + m;
    dr  = pr - 1.25*m;
    rr  = max(dr, br*0.95);
    alpha_p = inv_angle(br, pr);
    half_t  = 90/n - bl/(2*pr)*180/pi;
    ta      = 360/n;
    steps   = 16;

    pts = [for (ti = [0:n-1])
        let(base_ang = ti * ta)
        each concat(
            [[rr*cos(base_ang - ta/2 + 1), rr*sin(base_ang - ta/2 + 1)]],
            [for (s = [0:steps])
                let(r = rr + (ar - rr)*s/steps,
                    ia = inv_angle(br, r),
                    ang = base_ang + ia - alpha_p + half_t)
                [r*cos(ang), r*sin(ang)]
            ],
            [for (s = [0:steps])
                let(r = ar - (ar - rr)*s/steps,
                    ia = inv_angle(br, r),
                    ang = base_ang - (ia - alpha_p + half_t))
                [r*cos(ang), r*sin(ang)]
            ],
            [[rr*cos(base_ang + ta/2 - 1), rr*sin(base_ang + ta/2 - 1)]]
        )
    ];

    polygon(pts);
}

linear_extrude(height=8, center=true, convexity=10)
    ext_gear_2d(12, 2, 20, 0.5);
