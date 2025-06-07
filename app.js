const canvas = document.querySelector('canvas');
const instructions = document.getElementById('instructions');

const CPM = 152/2

// not sure how to resize the bauble player so just forcing 1080p
canvas.width = 1920;
canvas.height = 1080;

const bauble = new Bauble(canvas, {
  source: "#version 300 es\nprecision highp float;\n\nstruct Ray {\n  vec3 origin;\n  vec3 direction;\n};\nstruct PerspectiveCamera {\n  vec3 position;\n  vec3 direction;\n  vec3 up;\n  float fov;\n};\nstruct Light {\n  vec3 color;\n  vec3 direction;\n  float brightness;\n};\n\nout vec4 frag_color;\n\nuniform vec3 target;\nuniform float yscale;\nuniform float col1;\nuniform float yrot;\nuniform vec4 tracks;\nuniform vec3 pos;\nuniform float glitch;\nuniform float t;\nuniform vec4 viewport;\n\nmat2 rotation_2d(float angle) {\n  float s = sin(angle);\n  float c = cos(angle);\n  return mat2(c, s, -s, c);\n}\n\nfloat max_(vec2 v) {\n  return max(v.x, v.y);\n}\n\nmat3 cross_matrix(vec3 vec) {\n  return mat3(0.0, vec.z, -vec.y, -vec.z, 0.0, vec.x, vec.y, -vec.x, 0.0);\n}\n\nmat3 rotation_around(vec3 axis, float angle) {\n  return (cos(angle) * mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) + (sin(angle) * cross_matrix(axis)) + ((1.0 - cos(angle)) * outerProduct(axis, axis));\n}\n\nPerspectiveCamera roll(vec3 pos, float t, vec3 target) {\n  PerspectiveCamera camera = PerspectiveCamera(pos, normalize(target - pos), vec3(0.0, 1.0, 0.0), 45.0);\n  camera.up = rotation_around(camera.direction, sin(t) * 0.1) * camera.up;\n  return camera;\n}\n\nvec3 perspective_vector(float fov, vec2 frag_coord) {\n  float cot_half_fov = tan(radians(90.0 - (fov * 0.5)));\n  return normalize(vec3(frag_coord, cot_half_fov));\n}\n\nRay let_outer(vec2 frag_coord, vec3 pos, float t, vec3 target) {\n  {\n    PerspectiveCamera camera = roll(pos, t, target);\n    vec3 z_axis = camera.direction;\n    vec3 x_axis = normalize(cross(z_axis, camera.up));\n    vec3 y_axis = cross(x_axis, z_axis);\n    return Ray(camera.position, mat3(x_axis, y_axis, z_axis) * perspective_vector(camera.fov, frag_coord));\n  }\n}\n\nvec3 safe_div(vec3 a, vec3 b) {\n  return vec3((b.x == 0.0) ? 0.0 : (a.x / b.x), (b.y == 0.0) ? 0.0 : (a.y / b.y), (b.z == 0.0) ? 0.0 : (a.z / b.z));\n}\n\nmat3 rotation_y(float angle) {\n  float s = sin(angle);\n  float c = cos(angle);\n  return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);\n}\n\nmat3 rotation_z(float angle) {\n  float s = sin(angle);\n  float c = cos(angle);\n  return mat3(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0);\n}\n\nmat3 rotation_x(float angle) {\n  float s = sin(angle);\n  float c = cos(angle);\n  return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);\n}\n\nfloat max_1(vec3 v) {\n  return max(v.x, max(v.y, v.z));\n}\n\nfloat sdf_box(vec3 size, vec3 p) {\n  vec3 d = abs(p) - size;\n  return length(max(d, 0.0)) + min(max_1(d), 0.0);\n}\n\nfloat max_distance(vec3 p) {\n  float nearest = sdf_box(vec3(100.0, 100.0, 100.0), p);\n  nearest = max(nearest, -sdf_box(vec3(110.0, 33.0, 33.0), p));\n  return nearest;\n}\n\nfloat max_distance1(vec3 p) {\n  float nearest = max_distance(p);\n  nearest = max(nearest, -sdf_box(vec3(33.0, 110.0, 33.0), p));\n  return nearest;\n}\n\nfloat max_distance2(vec3 p) {\n  float nearest = max_distance1(p);\n  nearest = max(nearest, -sdf_box(vec3(33.0, 33.0, 110.0), p));\n  return nearest;\n}\n\nfloat rotate_outer(vec3 p) {\n  {\n    vec3 p1 = p * rotation_x(0.785398163397448);\n    return max_distance2(p1);\n  }\n}\n\nfloat rotate_outer1(vec3 p) {\n  {\n    vec3 p1 = p * rotation_z(0.628318530717959);\n    return rotate_outer(p1);\n  }\n}\n\nfloat scale_outer(vec3 factor, vec3 p) {\n  {\n    vec3 p1 = p / factor;\n    return rotate_outer1(p1);\n  }\n}\n\nfloat min_(vec3 v) {\n  return min(v.x, min(v.y, v.z));\n}\n\nfloat map_distance(vec3 factor, vec3 p) {\n  float dist = scale_outer(factor, p);\n  return min_(abs(factor)) * dist;\n}\n\nfloat let_outer1(vec3 p, float t, vec3 tile_index, float yscale) {\n  {\n    vec3 factor = (1.0 - vec3(0.0, 1.0, 0.0)) + (vec3(0.0, 1.0, 0.0) * (yscale * (cos((t + (tile_index.x * 100.0)) + tile_index.z) + 2.0)));\n    return map_distance(factor, p);\n  }\n}\n\nfloat rotate_outer2(vec3 p, float t, vec3 tile_index, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 p1 = p * rotation_y((p.y * 0.005) * (yrot / ((tracks.y * 2.0) + 0.3)));\n    return let_outer1(p1, t, tile_index, yscale);\n  }\n}\n\nfloat max_distance3(vec3 p, float t, vec3 tile_index, vec4 tracks, float yrot, float yscale) {\n  float nearest = rotate_outer2(p, t, tile_index, tracks, yrot, yscale);\n  nearest = max(nearest, dot(p, vec3(0.0, -1.0, 0.0)));\n  return nearest;\n}\n\nfloat move_outer(vec3 p, float t, vec3 tile_index) {\n  {\n    vec3 p1 = p - (vec3(0.0, 1000.0 * (sin(t + tile_index.x) + 1.0), 0.0) * 1.0);\n    return dot(p1, vec3(0.0, 1.0, 0.0));\n  }\n}\n\nfloat max_distance4(vec3 p, float t, vec3 tile_index, vec4 tracks, float yrot, float yscale) {\n  float nearest = max_distance3(p, t, tile_index, tracks, yrot, yscale);\n  nearest = max(nearest, move_outer(p, t, tile_index));\n  return nearest;\n}\n\nfloat with_outer(vec3 p, vec3 size, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 tile_index = round(safe_div(p, size));\n    vec3 p1 = p - (size * tile_index);\n    return max_distance4(p1, t, tile_index, tracks, yrot, yscale);\n  }\n}\n\nfloat let_outer2(vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 size = vec3(1500.0, 0.0, 1500.0);\n    return with_outer(p, size, t, tracks, yrot, yscale);\n  }\n}\n\nfloat min_distance(vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  float nearest = let_outer2(p, t, tracks, yrot, yscale);\n  nearest = min(nearest, dot(p, vec3(0.0, 1.0, 0.0)));\n  return nearest;\n}\n\nfloat nearest_distance(vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  return min_distance(p, t, tracks, yrot, yscale);\n}\n\nfloat march(out uint steps, Ray ray, float t, vec4 tracks, float yrot, float yscale) {\n  float ray_depth = 0.0;\n  for (steps = 0u; steps < 256u; ++steps) {\n    {\n      float depth = ray_depth;\n      vec3 P = ray.origin + (ray_depth * ray.direction);\n      vec3 p = P;\n      float dist = nearest_distance(p, t, tracks, yrot, yscale);\n      if (((dist >= 0.0) && (dist < 0.1)) || (ray_depth > 65536.0)) return ray_depth;\n      float rate = (dist > 0.0) ? 0.95 : 1.05;\n      ray_depth += dist * rate;\n      if (ray_depth < 0.0) return 0.0;\n    }\n  }\n  return ray_depth;\n}\n\nfloat with_outer1(vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).xyy * 0.005) + p;\n    return nearest_distance(p1, t, tracks, yrot, yscale);\n  }\n}\n\nfloat with_outer2(vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).yyx * 0.005) + p;\n    return nearest_distance(p1, t, tracks, yrot, yscale);\n  }\n}\n\nfloat with_outer3(vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).yxy * 0.005) + p;\n    return nearest_distance(p1, t, tracks, yrot, yscale);\n  }\n}\n\nfloat with_outer4(vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).xxx * 0.005) + p;\n    return nearest_distance(p1, t, tracks, yrot, yscale);\n  }\n}\n\nvec3 pow_(vec3 v, float e) {\n  return pow(v, vec3(e));\n}\n\nvec3 ok_slash_mix(vec3 from, vec3 to, float by) {\n  const mat3 to_lms_mat = mat3(0.412165612, 0.211859107, 0.0883097947, 0.536275208, 0.6807189584, 0.2818474174, 0.0514575653, 0.107406579, 0.6302613616);\n  const mat3 of_lms_mat = mat3(4.0767245293, -1.2681437731, -0.0041119885, -3.3072168827, 2.6093323231, -0.7034763098, 0.2307590544, -0.341134429, 1.7068625689);\n  vec3 from_lms = pow_(to_lms_mat * from, 1.0 / 3.0);\n  vec3 to_lms = pow_(to_lms_mat * to, 1.0 / 3.0);\n  vec3 lms = mix(from_lms, to_lms, by);\n  return of_lms_mat * (lms * lms * lms);\n}\n\nvec3 hsv(float hue, float saturation, float value) {\n  vec3 c = abs(mod((hue * 6.0) + vec3(0.0, 4.0, 2.0), 6.0) - 3.0);\n  return value * mix(vec3(1.0), clamp(c - 1.0, 0.0, 1.0), saturation);\n}\n\nfloat with_outer5(float depth, vec3 light_position, vec3 ray_dir, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 P = light_position + (ray_dir * depth);\n    vec3 p = P;\n    return nearest_distance(p, t, tracks, yrot, yscale);\n  }\n}\n\nLight cast_light_hard_shadow(vec3 light_color, vec3 light_position, vec3 P, vec3 normal, float t, vec4 tracks, float yrot, float yscale) {\n  if (light_position == P) return Light(light_color, vec3(0.0), 1.0);\n  vec3 to_light = normalize(light_position - P);\n  if (light_color == vec3(0.0)) return Light(light_color, to_light, 0.0);\n  if (dot(to_light, normal) < 0.0) return Light(light_color, to_light, 0.0);\n  vec3 target = (0.01 * normal) + P;\n  float light_distance = length(target - light_position);\n  vec3 ray_dir = (target - light_position) / light_distance;\n  float depth = 0.0;\n  for (uint i = 0u; i < 256u; ++i) {\n    float nearest = with_outer5(depth, light_position, ray_dir, t, tracks, yrot, yscale);\n    if (nearest < 0.01) break;\n    depth += nearest;\n  }\n  if (depth >= light_distance) return Light(light_color, to_light, 1.0);\n  else return Light(light_color, to_light, 0.0);\n}\n\nfloat with_outer6(float depth, vec3 light_position, vec3 ray_dir, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 P = light_position + (ray_dir * depth);\n    vec3 p = P;\n    return nearest_distance(p, t, tracks, yrot, yscale);\n  }\n}\n\nLight cast_light_soft_shadow(vec3 light_color, vec3 light_position, float softness, vec3 P, vec3 normal, float t, vec4 tracks, float yrot, float yscale) {\n  if (softness == 0.0) return cast_light_hard_shadow(light_color, light_position, P, normal, t, tracks, yrot, yscale);\n  if (light_position == P) return Light(light_color, vec3(0.0), 1.0);\n  vec3 to_light = normalize(light_position - P);\n  if (light_color == vec3(0.0)) return Light(light_color, to_light, 0.0);\n  if (dot(to_light, normal) < 0.0) return Light(light_color, to_light, 0.0);\n  vec3 target = (0.01 * normal) + P;\n  float light_distance = length(target - light_position);\n  vec3 ray_dir = (target - light_position) / light_distance;\n  float brightness = 1.0;\n  float sharpness = 1.0 / (softness * softness);\n  float last_nearest = 1000000.0;\n  float depth = 0.0;\n  for (uint i = 0u; i < 256u; ++i) {\n    float nearest = with_outer6(depth, light_position, ray_dir, t, tracks, yrot, yscale);\n    if (nearest < 0.01) break;\n    float intersect_offset = (nearest * nearest) / (2.0 * last_nearest);\n    float intersect_distance = sqrt((nearest * nearest) - (intersect_offset * intersect_offset));\n    brightness = min(brightness, (sharpness * intersect_distance) / max(0.0, (light_distance - depth) - intersect_offset));\n    depth += nearest;\n    last_nearest = nearest;\n  }\n  if (depth >= light_distance) return Light(light_color, to_light, brightness);\n  else return Light(light_color, to_light, 0.0);\n}\n\nfloat with_outer7(vec3 P, uint i, vec3 step, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 P1 = (float(i) * step) + P;\n    vec3 p = P1;\n    return max(nearest_distance(p, t, tracks, yrot, yscale), 0.0);\n  }\n}\n\nfloat calculate_occlusion(uint step_count, float max_distance, vec3 dir, vec3 P, vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  float step_size = max_distance / float(step_count);\n  float baseline = nearest_distance(p, t, tracks, yrot, yscale);\n  float occlusion = 0.0;\n  vec3 step = dir * step_size;\n  for (uint i = 1u; i <= step_count; ++i) {\n    float expected_distance = (float(i) * step_size) + baseline;\n    float actual_distance = with_outer7(P, i, step, t, tracks, yrot, yscale);\n    occlusion += actual_distance / expected_distance;\n  }\n  return clamp(occlusion / float(step_count), 0.0, 1.0);\n}\n\nvec3 normalize_safe(vec3 v) {\n  return (v == vec3(0.0)) ? v : normalize(v);\n}\n\nLight cast_light_no_shadow(vec3 light_color, vec3 light_position, vec3 P) {\n  return Light(light_color, normalize_safe(light_position - P), 1.0);\n}\n\nLight do_(vec3 P, vec3 normal, float occlusion) {\n  Light light = cast_light_no_shadow(vec3(0.15), P + (normal * 0.1), P);\n  light.brightness = light.brightness * mix(0.1, 1.0, occlusion);\n  return light;\n}\n\nuint union_color_index(vec3 p, float t, vec4 tracks, float yrot, float yscale) {\n  float nearest = dot(p, vec3(0.0, 1.0, 0.0));\n  uint nearest_index = 0u;\n  float d = dot(p, vec3(0.0, 1.0, 0.0));\n  if (d < 0.0) return 0u;\n  if (d < nearest) {\n    nearest = d;\n    nearest_index = 0u;\n  }\n  float d1 = let_outer2(p, t, tracks, yrot, yscale);\n  if (d1 < 0.0) return 1u;\n  if (d1 < nearest) {\n    nearest = d1;\n    nearest_index = 1u;\n  }\n  return nearest_index;\n}\n\nvec3 mod289(vec3 x) {\n  return x - (floor(x * (1.0 / 289.0)) * 289.0);\n}\n\nvec4 mod2891(vec4 x) {\n  return x - (floor(x * (1.0 / 289.0)) * 289.0);\n}\n\nvec4 permute(vec4 x) {\n  return mod2891(((x * 34.0) + 10.0) * x);\n}\n\nvec4 taylor_inv_sqrt(vec4 r) {\n  return 1.79284291400159 - (0.85373472095314 * r);\n}\n\nvec3 fade(vec3 t) {\n  return t * t * t * ((t * ((t * 6.0) - 15.0)) + 10.0);\n}\n\nfloat perlin(vec3 point) {\n  vec3 Pi0 = floor(point);\n  vec3 Pi1 = Pi0 + 1.0;\n  vec3 Pi01 = mod289(Pi0);\n  vec3 Pi11 = mod289(Pi1);\n  vec3 Pf0 = fract(point);\n  vec3 Pf1 = Pf0 - 1.0;\n  vec4 ix = vec4(Pi01.x, Pi11.x, Pi01.x, Pi11.x);\n  vec4 iy = vec4(Pi01.yy, Pi11.yy);\n  vec4 iz0 = Pi01.zzzz;\n  vec4 iz1 = Pi11.zzzz;\n  vec4 ixy = permute(permute(ix) + iy);\n  vec4 ixy0 = permute(ixy + iz0);\n  vec4 ixy1 = permute(ixy + iz1);\n  vec4 gx0 = ixy0 * (1.0 / 7.0);\n  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;\n  gx0 = fract(gx0);\n  vec4 gz0 = 0.5 - abs(gx0) - abs(gy0);\n  vec4 sz0 = step(gz0, vec4(0.0));\n  gx0 -= sz0 * (step(0.0, gx0) - 0.5);\n  gy0 -= sz0 * (step(0.0, gy0) - 0.5);\n  vec4 gx1 = ixy1 * (1.0 / 7.0);\n  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;\n  gx1 = fract(gx1);\n  vec4 gz1 = 0.5 - abs(gx1) - abs(gy1);\n  vec4 sz1 = step(gz1, vec4(0.0));\n  gx1 -= sz1 * (step(0.0, gx1) - 0.5);\n  gy1 -= sz1 * (step(0.0, gy1) - 0.5);\n  vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);\n  vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);\n  vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);\n  vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);\n  vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);\n  vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);\n  vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);\n  vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);\n  vec4 norm0 = taylor_inv_sqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));\n  g000 *= norm0.x;\n  g010 *= norm0.y;\n  g100 *= norm0.z;\n  g110 *= norm0.w;\n  vec4 norm1 = taylor_inv_sqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));\n  g001 *= norm1.x;\n  g011 *= norm1.y;\n  g101 *= norm1.z;\n  g111 *= norm1.w;\n  float n000 = dot(g000, Pf0);\n  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));\n  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));\n  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));\n  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));\n  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));\n  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));\n  float n111 = dot(g111, Pf1);\n  vec3 fade_xyz = fade(Pf0);\n  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);\n  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);\n  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);\n  return 2.2 * n_xyz;\n}\n\nfloat with_outer8(vec3 p) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).xyy * 0.005) + p;\n    return perlin(p1 / 10.0);\n  }\n}\n\nfloat with_outer9(vec3 p) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).yyx * 0.005) + p;\n    return perlin(p1 / 10.0);\n  }\n}\n\nfloat with_outer10(vec3 p) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).yxy * 0.005) + p;\n    return perlin(p1 / 10.0);\n  }\n}\n\nfloat with_outer11(vec3 p) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).xxx * 0.005) + p;\n    return perlin(p1 / 10.0);\n  }\n}\n\nvec3 blinn_phong(Light light, vec3 color, float shininess, float glossiness, vec3 normal, Ray ray) {\n  if (light.direction == vec3(0.0)) return color * light.color * light.brightness;\n  vec3 halfway_dir = normalize(light.direction - ray.direction);\n  float specular_strength = shininess * pow(max(dot(normal, halfway_dir), 0.0), glossiness * glossiness);\n  float diffuse = max(0.0, dot(normal, light.direction));\n  return ((light.color * light.brightness) * specular_strength) + (color * diffuse * light.color * light.brightness);\n}\n\nvec3 shade(Light light, Light light1, vec3 normal, Ray ray, vec3 temp) {\n  vec3 result = vec3(0.0);\n  result += blinn_phong(light1, temp, 0.25, 10.0, normal, ray);\n  result += blinn_phong(light, temp, 0.25, 10.0, normal, ray);\n  return result;\n}\n\nvec3 shade_outer(float col1, Light light, Light light1, vec3 normal, Ray ray) {\n  {\n    vec3 temp = ok_slash_mix(hsv(0.0833333333333333, 0.98, 1.0), hsv(0.666666666666667, 0.98, 1.0), col1);\n    return shade(light1, light, normal, ray, temp);\n  }\n}\n\nfloat fresnel(float exponent, vec3 normal, Ray ray) {\n  return pow(1.0 + dot(normal, ray.direction), exponent);\n}\n\nvec3 map_color(float col1, Light light, Light light1, vec3 normal, Ray ray) {\n  vec3 color = shade_outer(col1, light, light1, normal, ray);\n  return color + (ok_slash_mix(hsv(0.666666666666667, 0.98, 1.0), hsv(0.583333333333333, 0.98, 1.0), col1) * fresnel(1.0, normal, ray));\n}\n\nvec3 with_outer12(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray) {\n  {\n    vec3 normal1 = normalize(normal - ((normalize((vec2(1.0, -1.0).xyy * with_outer8(p)) + (vec2(1.0, -1.0).yyx * with_outer9(p)) + (vec2(1.0, -1.0).yxy * with_outer10(p)) + (vec2(1.0, -1.0).xxx * with_outer11(p))) * 1.0) * perlin(p / 10.0)));\n    return map_color(col1, light, light1, normal1, ray);\n  }\n}\n\nfloat with_outer13(vec3 p) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).xyy * 0.005) + p;\n    return perlin(p1 / 10.0);\n  }\n}\n\nfloat with_outer14(vec3 p) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).yyx * 0.005) + p;\n    return perlin(p1 / 10.0);\n  }\n}\n\nfloat with_outer15(vec3 p) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).yxy * 0.005) + p;\n    return perlin(p1 / 10.0);\n  }\n}\n\nfloat with_outer16(vec3 p) {\n  {\n    vec3 p1 = (vec2(1.0, -1.0).xxx * 0.005) + p;\n    return perlin(p1 / 10.0);\n  }\n}\n\nvec3 shade1(Light light, Light light1, vec3 normal, Ray ray, vec3 temp) {\n  vec3 result = vec3(0.0);\n  result += blinn_phong(light1, temp, 0.3, 5.0, normal, ray);\n  result += blinn_phong(light, temp, 0.3, 5.0, normal, ray);\n  return result;\n}\n\nvec3 shade_outer1(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray) {\n  {\n    vec3 temp = ok_slash_mix(hsv(0.0, 0.98, 1.0), ok_slash_mix(hsv(0.0833333333333333, 0.98, 1.0), hsv(0.666666666666667, 0.98, 1.0), col1), ((p.y / 200.0) + 1.0) * 0.5);\n    return shade1(light, light1, normal, ray, temp);\n  }\n}\n\nvec3 map_color1(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray) {\n  vec3 color = shade_outer1(col1, light, light1, normal, p, ray);\n  return color + (hsv(0.666666666666667, 0.98, 1.0) * fresnel(1.4, normal, ray));\n}\n\nvec3 rotate_outer3(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray) {\n  {\n    vec3 p1 = p * rotation_x(0.785398163397448);\n    return map_color1(col1, light, light1, normal, p1, ray);\n  }\n}\n\nvec3 rotate_outer4(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray) {\n  {\n    vec3 p1 = p * rotation_z(0.628318530717959);\n    return rotate_outer3(col1, light, light1, normal, p1, ray);\n  }\n}\n\nvec3 scale_outer1(float col1, vec3 factor, Light light, Light light1, vec3 normal, vec3 p, Ray ray) {\n  {\n    vec3 p1 = p / factor;\n    return rotate_outer4(col1, light, light1, normal, p1, ray);\n  }\n}\n\nvec3 let_outer3(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray, float t, vec3 tile_index, float yscale) {\n  {\n    vec3 factor = (1.0 - vec3(0.0, 1.0, 0.0)) + (vec3(0.0, 1.0, 0.0) * (yscale * (cos((t + (tile_index.x * 100.0)) + tile_index.z) + 2.0)));\n    return scale_outer1(col1, factor, light1, light, normal, p, ray);\n  }\n}\n\nvec3 rotate_outer5(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray, float t, vec3 tile_index, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 p1 = p * rotation_y((p.y * 0.005) * (yrot / ((tracks.y * 2.0) + 0.3)));\n    return let_outer3(col1, light, light1, normal, p1, ray, t, tile_index, yscale);\n  }\n}\n\nvec3 with_outer17(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray, float t, vec3 tile_index, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 normal1 = normalize(normal - ((normalize((vec2(1.0, -1.0).xyy * with_outer13(p)) + (vec2(1.0, -1.0).yyx * with_outer14(p)) + (vec2(1.0, -1.0).yxy * with_outer15(p)) + (vec2(1.0, -1.0).xxx * with_outer16(p))) * 1.0) * perlin(p / 10.0)));\n    return rotate_outer5(col1, light, light1, normal1, p, ray, t, tile_index, tracks, yrot, yscale);\n  }\n}\n\nvec3 with_outer18(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray, vec3 size, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 tile_index = round(safe_div(p, size));\n    vec3 p1 = p - (size * tile_index);\n    return with_outer17(col1, light, light1, normal, p1, ray, t, tile_index, tracks, yrot, yscale);\n  }\n}\n\nvec3 let_outer4(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    vec3 size = vec3(1500.0, 0.0, 1500.0);\n    return with_outer18(col1, light1, light, normal, p, ray, size, t, tracks, yrot, yscale);\n  }\n}\n\nvec3 union_color(float col1, Light light, Light light1, vec3 normal, vec3 p, Ray ray, float t, vec4 tracks, float yrot, float yscale) {\n  switch (union_color_index(p, t, tracks, yrot, yscale)) {\n  case 0u: return with_outer12(col1, light1, light, normal, p, ray);\n  case 1u: return let_outer4(col1, light, light1, normal, p, ray, t, tracks, yrot, yscale);\n  }\n  return vec3(0.0, 0.0, 0.0);\n}\n\nvec3 hoist_outer(vec3 P, float col1, vec3 normal, vec3 p, Ray ray, float t, vec4 tracks, float yrot, float yscale) {\n  {\n    Light light = cast_light_soft_shadow(vec3(1.15), P - (normalize(vec3(-2.0, -2.0, -1.0)) * 2048.0), 0.25, P, normal, t, tracks, yrot, yscale);\n    float occlusion = calculate_occlusion(8u, 20.0, normal, P, p, t, tracks, yrot, yscale);\n    Light light1 = do_(P, normal, occlusion);\n    return union_color(col1, light1, light, normal, p, ray, t, tracks, yrot, yscale);\n  }\n}\n\nvec4 sample_(float col1, vec2 frag_coord, vec3 pos, float t, vec3 target, vec4 tracks, float yrot, float yscale) {\n  Ray ray_star = Ray(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));\n  vec3 ortho_quad = vec3(0.0, 0.0, 0.0);\n  float ortho_scale = 0.0;\n  float fov = 0.0;\n  ray_star = let_outer(frag_coord, pos, t, target);\n  uint steps = 0u;\n  {\n    Ray ray = ray_star;\n    float depth = march(steps, ray, t, tracks, yrot, yscale);\n    vec3 P = ray.origin + (ray.direction * depth);\n    vec3 p = P;\n    float dist = nearest_distance(p, t, tracks, yrot, yscale);\n    vec3 normal = normalize((vec2(1.0, -1.0).xyy * with_outer1(p, t, tracks, yrot, yscale)) + (vec2(1.0, -1.0).yyx * with_outer2(p, t, tracks, yrot, yscale)) + (vec2(1.0, -1.0).yxy * with_outer3(p, t, tracks, yrot, yscale)) + (vec2(1.0, -1.0).xxx * with_outer4(p, t, tracks, yrot, yscale)));\n    vec4 color = vec4(0.0);\n    color = (dist >= 10.0) ? vec4(ok_slash_mix(hsv(0.0833333333333333, 0.98, 1.0), hsv(0.583333333333333, 0.98, 1.0), col1), 1.0) : vec4(hoist_outer(P, col1, normal, p, ray, t, tracks, yrot, yscale), 1.0);\n    return color;\n  }\n}\n\nvoid main() {\n  const float gamma = 2.2;\n  vec3 color = vec3(0.0, 0.0, 0.0);\n  float alpha = 0.0;\n  const uint aa_grid_size = 1u;\n  const float aa_sample_width = 1.0 / float(1u + aa_grid_size);\n  const vec2 pixel_origin = vec2(0.5, 0.5);\n  vec2 local_frag_coord = gl_FragCoord.xy - viewport.xy;\n  mat2 rotation = rotation_2d(0.2);\n  for (uint y = 1u; y <= aa_grid_size; ++y) {\n    for (uint x = 1u; x <= aa_grid_size; ++x) {\n      vec2 sample_offset = (aa_sample_width * vec2(float(x), float(y))) - pixel_origin;\n      sample_offset = rotation * sample_offset;\n      sample_offset = fract(sample_offset + pixel_origin) - pixel_origin;\n      {\n        vec2 Frag_Coord = local_frag_coord + sample_offset;\n        vec2 resolution = viewport.zw;\n        vec2 frag_coord = ((Frag_Coord - (0.5 * resolution)) / max_(resolution)) * 2.0;\n        vec4 this_sample = clamp(sample_(col1, frag_coord, pos, t, target, tracks, yrot, yscale), 0.0, 1.0);\n        color += this_sample.rgb * this_sample.a;\n        alpha += this_sample.a;\n      }\n    }\n  }\n  if (alpha > 0.0) {\n    color = color / alpha;\n    alpha /= float(aa_grid_size * aa_grid_size);\n  }\n  frag_color = vec4(pow_(color, 1.0 / gamma), alpha);\n}\n",
  animation: true,
  uniforms: {
    target: "vec3",
    yscale: "float",
    col1: "float",
    yrot: "float",
    tracks: "vec4",
    pos: "vec3",
    glitch: "float"
  }
});
bauble.set({
  target: [0, 0, 0],
  yscale: 3,
  col1: 1,
  yrot: 1,
  tracks: [0, 0, 99, 99],
  pos: [0, 100, 600],
  glitch: 0
});

let demoStart = 0;
let isPlaying = false;

const tracks = {
  bd: 99,
  sd: 99,
}

const pingTrack = (name) => (x) => {
  tracks[name] = (Date.now() - demoStart) / 1000;
  // console.log(`Track ${name} pinged`, tracks[name]);
  return x
}

let credits = false;
const update = () => {
  if (!isPlaying) {
    return;
  }

  const now = Date.now() - demoStart;
  const t = now / 1000;
  const c = t * CPM / 60;
  const bar = c / 8;

  let target = [0, 0, 0];
  let pos = [0, 0, 0];
  let yrot = 0;
  let yscale = 1;
  let col1 = 0;

  if (bar % 4 < 2) {
    if (bar % 2 < 1) {
      pos = [Math.sin(t * 0.5) * 2000, Math.cos(c * 0.3) * 500 + 600, Math.cos(t * 0.5) * 2000];
    } else {
      pos =    [1000, 3000 + Math.sin(t), t * 2000]
      target = [1000, 2970 + Math.sin(t + 10) * 0.1, t * 2000 + 40];
    }
  } else {
    pos = [t*1000 + Math.sin(t * 0.2) * 3000, Math.cos(t * 0.3) * 500 + 600, Math.cos(t * 0.4) * 3000]
    target = [t*1000 + Math.sin(t * 0.2) * 500, Math.cos(t * 0.3) * 100 + 100, Math.cos(t * 0.4) * -500]
  }

  if (bar < 1) {
    yrot = 0.1
    col1 = 0;
  } else if (bar < 5) {
    yrot = 0.25
    col1 = 0;
  } else if (bar < 6) {
    yrot = 0.8
    col1 = 0.5;
  } else if (bar < 12) {
    yrot = 1.6
    col1 = 1.0;
  }

  if (bar > 10 && !credits) {
    document.getElementById("credits").style.display = "flex";
    credits = true;
  }

  if (bar > 12) {
    music.pause();
    isPlaying = false;
  }

  bauble.set({
    t,
    yscale: Math.sin(t * 0.5 - 1.6) * yscale + (yscale + 1),
    yrot,
    pos,
    target,
    tracks: [t - tracks.bd, t - tracks.sd, 99, 99],
    col1,
  });

  requestAnimationFrame(update);
}

initStrudel({
  prebake: () => {
    samples('strudel.json')
  }
});

await new Promise(resolve => setTimeout(resolve, 100));

const amen = sound("amen").bpf("1000").room(.3).rsize(4).gain(1.0);
const bd = sound(`<[bd ~ ~ bd ~ ~ ~ ~ bd ~ ~ bd ~ bd ~ ~]>`)
const sd = sound(`<[~ sd] * 2>`).when(`<[0  1] * 2>`, pingTrack("sd"))
const rd = sound(`<[~ rd] * 4>`)
const hh = sound(`<[~ hh hh hh ~ ~ ~ ~] * 2>`)
const bass = note(`<[d1@3 [f1 a1] d#1@4] / 4>`)
  .add(note("0,.05"))
  .sound("triangle")
  .lpf("400 600 800 600 400 600 800 1000")
  .lpq("5").lpenv(2)
  .hpf("80")
  .adsr(".0:.1:1.0:.2")
  .jux(press)
  .phaser(4).gain(0.5)
const arpBass = note(`<
    [~ d1 d1 d2 ~ d1 f2 d1] * 2
  >`)
  .add(note("0,.2"))
  .sound("sawtooth")
  .lpf("<300>").lpq("10").lpenv(4)
  .adsr(".0:.1:.0:.2")
const chords = note("<[d2, f3, a4] [d#2, g3, a#4]> / 2")
    .add(note("0,.05"))
    .sound("sawtooth")
    .adsr(".0:.1:.0:4.")
    .lpf("1400 1800 1200 2000")
    .lpenv(4)
    .delay(0.5).room(1.5).rsize(4)
    .gain(0.5)
const arp = note("<[[~ c5 g5 d5]*4 [~ d5 g5 a#4]*4]/2>").sometimes(x => x.sub("7"))
  .sound("sawtooth")
  .adsr(".0:.1:.0:1.")
  .hpf("1300")
  .lpf("1800").lpenv(4)
  .delay(0.4).room(1.5).rsize(4).gain(0.32)
const lead = note("c6(3,8,0)")
    .add(note("0,.05"))
    .sound("square")
    .adsr(".0:.1:.0:2.")
    .hpf("1100")
    .lpf("[1400 1900 2200 1400 1100 1400 1600 2200]/4")
    .lpq("5")
    .lpenv(4)
    .delay(0.4).room(1.5).rsize(3).gain(0.2)

const music = stack(
  amen.   mask("[0 0 0 0  0 1 1 1  1 1 0 0  0 0 0 0] / 64"),
  bd.     mask("[0 1 1 1  0 0 1 1  1 1 1 1  0 0 0 0] / 64"),
  sd.     mask("[1 1 1 1  0 0 1 1  1 1 1 1  0 0 0 0] / 64"),
  rd.     mask("[1 1 1 1  0 0 1 1  1 1 1 1  0 0 0 0] / 64"),
  hh.     mask("[1 1 1 1  0 0 1 1  1 1 1 1  0 0 0 0] / 64"),
  arpBass.mask("[1 1 1 1  0 0 0 0  1 1 1 1  0 0 0 0] / 64"),
  bass.   mask("[0 0 0 0  1 1 1 1  0 0 0 0  0 0 0 0] / 64"),
  chords. mask("[0 0 0 1  1 1 1 1  1 1 0 0  0 0 0 0] / 64"),
  arp.    mask("[0 1 1 1  1 1 1 1  1 1 1 1  0 0 0 0] / 64"),
  lead.   mask("[0 0 0 0  0 0 1 1  1 1 1 1  0 0 0 0] / 64"),
  
  ).cpm(CPM)


let playingMusic = false;
const startMusic = () => {
  playingMusic = true;
  music.play()
}

document.addEventListener('keydown', async (e) => {
    if (e.key === 'f') {
        if (!document.fullscreenElement) {
            canvas.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    if (e.key === ' ') {
        document.getElementById("warning").style.display = 'block';
        await new Promise(resolve => setTimeout(resolve, 0));
        bauble.togglePlay();
        instructions.style.display = 'none';
        canvas.style.display = 'block';
        demoStart = Date.now();
        isPlaying = true;
        update();
        startMusic();
    }
});
