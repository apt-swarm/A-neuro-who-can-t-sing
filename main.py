# python -m pygbag --PYBUILD 3.12 --ume_block 0 --template noctx.tmpl .

# /// script
# dependencies = [
#  "numpy",
#  "pygame",
#  "struct",
#  "zengl",
#  "marshmallow",
#  "opencv-python"
# ]
# ///

import numpy as np
import asyncio
import pygame
import pygame.mixer as mixer
import struct
import zengl
from PIL import Image

HEIGHT, WIDTH = 1080, 1920
FPS = 60

pygame.init()
mixer.init()

SFXchannel = mixer.Channel(0)
musicChannel = mixer.Channel(1)
dialogueChannel = mixer.Channel(2)

trueBuzzer = mixer.Sound("sfx/buzzers/correct.ogg")
wrongBuzzer = mixer.Sound("sfx/buzzers/wrong.ogg")
crowdNoise = mixer.Sound("sfx/crowdSFX.ogg")
introDialogue = mixer.Sound("sfx/dialogue/IntroDialogue.ogg")
game1Dialogue1 = mixer.Sound("sfx/dialogue/Game1Dialogue1.ogg")
game1Dialogue2 = mixer.Sound("sfx/dialogue/Game1Dialogue2.ogg")
game1Dialogue3 = mixer.Sound("sfx/dialogue/Game1Dialogue3.ogg")
game2Dialogue1 = mixer.Sound("sfx/dialogue/Game2Dialogue1.ogg")
game2Dialogue2 = mixer.Sound("sfx/dialogue/Game2Dialogue2.ogg")
game2Dialogue3 = mixer.Sound("sfx/dialogue/Game2Dialogue3.ogg")
game3Dialogue1 = mixer.Sound("sfx/dialogue/Game3Dialogue1.ogg")
game3Dialogue2 = mixer.Sound("sfx/dialogue/Game3Dialogue2.ogg")
FinalDialogue = mixer.Sound("sfx/dialogue/FinalDialogue.ogg")

POGsong = mixer.Sound("sfx/POGsong.ogg")
mainMenuMusic = False
paranoidMusic = mixer.Sound("sfx/BGmusic/NeuroParanoidMusic.ogg")
hopefulMusic = mixer.Sound("sfx/BGmusic/NeuroHopefulMusic.ogg")
pianoMusic = mixer.Sound("sfx/BGmusic/NeuroBGMpiano.ogg")
musicboxMusic = mixer.Sound("sfx/BGmusic/NeuroBGMmusicbox.ogg")
titlescreenMusic = mixer.Sound("sfx/BGmusic/NeuroTitleScreenMusic.ogg")

screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.OPENGL|pygame.DOUBLEBUF)
clock = pygame.time.Clock()
ctx = zengl.context()
size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples= 4)
depth = ctx.image(size, 'depth24plus', samples= 4)
output = ctx.image(size, 'rgba8unorm')

#####################################################################################

input_map = {'right': pygame.K_d,
             'left': pygame.K_a,
             'forwards': pygame.K_w,
             'backwards': pygame.K_s,
             'jump': pygame.K_SPACE,
             'sprint': pygame.K_LSHIFT,
             'escape': pygame.K_p,
             'interact': pygame.K_e,
             'leftArrow': pygame.K_RIGHT,
             'rightArrow': pygame.K_LEFT}

#####################################################################################

CONTINUE = 0
NEW_GAME = 1
OPEN_MENU = 2
EXIT = 3
CREDITS = 4

SCENES = {'anim0': 0,
          "beforeGame1": 1,
          "vedals room1": 2,
          "minigame1": 3,
          "afterGame1": 4,
          "beforeGame2": 5,
          "vedals room2": 6,
          "minigame2": 7,
          "afterGame2": 8,
          "beforeGame3": 9,
          "vedals room3": 10,
          "minigame3": 11,
          "afterGame3": 12,
          "anim1": 13,
          "credits": 14}

ENTITY_TYPE = {"player": 0,
               "bounding_box" : 1,
               "keyUI": 2,
               "room": 3,
               "bed": 4,
               "stool": 5,
               "desk": 6,
               "chair": 7,
               "laptop": 8,
               "tablet": 9
               }

#####################################################################################

#pyrr functions that i copied cuz import pyrr causes long load times in browsers

def create_perspective_projection_from_bounds(left,right,bottom,top,near,far,dtype=None):
    A = (right + left) / (right - left)
    B = (top + bottom) / (top - bottom)
    C = -(far + near) / (far - near)
    D = -2. * far * near / (far - near)
    E = 2. * near / (right - left)
    F = 2. * near / (top - bottom)

    return np.array(((E,  0., 0., 0.),
                     (0., F,  0., 0.),
                     (A,  B,  C, -1.),
                     (0., 0., D,  0.))
    )
def normalize(vec):

    return (vec.T  / np.sqrt(np.sum(vec**2,axis=-1))).T
def create_from_eulers(eulers, dtype=None):
    dtype = dtype or eulers.dtype

    roll, pitch, yaw = eulers

    sP = np.sin(pitch)
    cP = np.cos(pitch)
    sR = np.sin(roll)
    cR = np.cos(roll)
    sY = np.sin(yaw)
    cY = np.cos(yaw)

    return np.array(
        [
            # m1
            [
                cY * cP,
                -cY * sP * cR + sY * sR,
                cY * sP * sR + sY * cR,
            ],
            # m2
            [
                sP,
                cP * cR,
                -cP * sR,
            ],
            # m3
            [
                -sY * cP,
                sY * sP * cR + cY * sR,
                -sY * sP * sR + cY * cR,
            ]
        ],
        dtype=dtype
    )
def create_from_quaternion(quat, dtype=None):
    dtype = dtype

    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    sqw = qw**2
    sqx = qx**2
    sqy = qy**2
    sqz = qz**2
    qxy = qx * qy
    qzw = qz * qw
    qxz = qx * qz
    qyw = qy * qw
    qyz = qy * qz
    qxw = qx * qw

    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = ( sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs
    m10 = 2.0 * (qxy + qzw) * invs
    m01 = 2.0 * (qxy - qzw) * invs
    m20 = 2.0 * (qxz - qyw) * invs
    m02 = 2.0 * (qxz + qyw) * invs
    m21 = 2.0 * (qyz + qxw) * invs
    m12 = 2.0 * (qyz - qxw) * invs

    return np.array([
        [m00, m01, m02, 0],
        [m10, m11, m12, 0],
        [m20, m21, m22, 0],
        [0,   0,   0,   1]
    ], dtype=dtype)
def create_from_translation(vec, dtype=None):

    dtype = dtype
    mat = np.identity(4, dtype=dtype)
    mat[3, 0:3] = vec[:3]
    return mat
def create_from_scale(scale, dtype=None):
    m = np.diagflat([scale[0], scale[1], scale[2], 1.0])
    if dtype:
        m = m.astype(dtype)
    return m
def ray_intersect_aabb(ray, aabb):

    direction = ray[1]
    dir_fraction = np.empty(3, dtype = ray.dtype)
    dir_fraction[direction == 0.0] = np.inf
    dir_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])

    t1 = (aabb[0,0] - ray[0,0]) * dir_fraction[ 0 ]
    t2 = (aabb[1,0] - ray[0,0]) * dir_fraction[ 0 ]
    t3 = (aabb[0,1] - ray[0,1]) * dir_fraction[ 1 ]
    t4 = (aabb[1,1] - ray[0,1]) * dir_fraction[ 1 ]
    t5 = (aabb[0,2] - ray[0,2]) * dir_fraction[ 2 ]
    t6 = (aabb[1,2] - ray[0,2]) * dir_fraction[ 2 ]


    tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
    tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))

    # if tmax < 0, ray (line) is intersecting AABB
    # but the whole AABB is behind the ray start
    if tmax < 0:
        return None

    # if tmin > tmax, ray doesn't intersect AABB
    if tmin > tmax:
        return None

    # t is the distance from the ray point
    # to intersection

    t = min(x for x in [tmin, tmax] if x >= 0)
    point = ray[0] + (ray[1] * t)
    return point

projection = create_perspective_projection_from_bounds(-0.1, 0.1, -0.1*HEIGHT/WIDTH, 0.1*HEIGHT/WIDTH, 0.1, 2000)

def shader2D(vertexBuffer, texBuffer, texture):

    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;

            layout(location = 0) in vec2 vpos;
            layout(location = 1) in vec2 vtex;

            out vec2 TexCoords;

            void main()
            {
                TexCoords = vtex;
                gl_Position = vec4(vpos, 0, 1);
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;

            in vec2 TexCoords;

            uniform sampler2D material;

            layout(location = 0) out vec4 color;

            void main()
            {
                color = texture(material, TexCoords);
                color = pow(color, vec4(0.45));
            }
        """,

        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],

        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "2f", 0),
                         *zengl.bind(ctx.buffer(texBuffer), "2f", 1)],

        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )
def shader2Danitex(vertexBuffer, texBuffer, texture, frameAmount):

    texBuffer /= [frameAmount, 1]

    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;

            layout(location = 0) in vec2 vpos;
            layout(location = 1) in vec2 vtex;
            uniform float ofset;

            out vec2 TexCoords;

            void main()
            {
                TexCoords = vtex + vec2(ofset, 0);
                gl_Position = vec4(vpos, 0, 1);
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;

            in vec2 TexCoords;
            uniform sampler2D material;

            layout(location = 0) out vec4 color;

            void main()
            {
                color = texture(material, TexCoords);
                color = pow(color, vec4(0.45));
            }
        """,

        uniforms={'ofset': 0},

        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],

        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "2f", 0),
                         *zengl.bind(ctx.buffer(texBuffer), "2f", 1)],

        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )
def shader3D(vertexBuffer, normBuffer, texBuffer, texture):

    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;

            layout(location = 0) in vec3 vpos;
            layout(location = 1) in vec3 vnorm;
            layout(location = 2) in vec2 vtex;

            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;

            out vec2 TexCoords;
            out vec3 fragPos;
            out vec3 fragNorm;

            void main()
            {
                vec4 vertPos = model * vec4(vpos, 1.0);

                TexCoords = vtex;
                fragPos = vertPos.xyz;
                fragNorm = (model * vec4(vnorm, 0)).xyz;
                gl_Position = projection * view * vertPos;
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;

            in vec2 TexCoords;
            in vec3 fragPos;
            in vec3 fragNorm;

            uniform sampler2D material;
            uniform vec3 camPos;
            uniform vec3 lightposition[1];
            uniform vec3 lightcolor[1];
            uniform float lightstrength[1];

            layout(location = 0) out vec4 color;

            vec3 calcPointlight(int i)
            {
                vec3 baseTexture = texture(material, TexCoords).rgb;
                vec3 result = vec3(0);

                vec3 relLightPos = lightposition[i] - fragPos;
                float distance = length(relLightPos);
                relLightPos = normalize(relLightPos);

                vec3 relCamPos = normalize(camPos - fragPos);
                vec3 halfVec = normalize(relLightPos + relCamPos);

                result += lightcolor[i] * lightstrength[i] * max(0.0, dot(fragNorm, relLightPos)) / (distance * distance) * baseTexture; //diffuse
                result += lightcolor[i] * lightstrength[i] * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / (distance * distance); //specular

                return result;
            }

            void main()
            {
                vec4 baseTex = texture(material, TexCoords);
                vec3 temp = 0.2 * baseTex.rgb; //ambient
                temp += calcPointlight(0);

                color = pow(vec4(temp, baseTex.a), vec4(0.45));
            }
        """,

        uniforms={'projection': projection.flatten(), 'view': np.identity(4).flatten(), 'model': np.identity(4).flatten(),
                  'camPos' : [0,0,0],
                  'lightposition': [[0, 1000, 0]],
                  'lightcolor': [[255,255,255]],
                  'lightstrength': [500]},

        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],

        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                         *zengl.bind(ctx.buffer(normBuffer), "3f", 1),
                         *zengl.bind(ctx.buffer(texBuffer), "2f", 2)],

        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )
def shader3Danitex(vertexBuffer, normBuffer, texBuffer, texture, frameAmount):

    texBuffer /= [frameAmount, 1]

    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;

            layout(location = 0) in vec3 vpos;
            layout(location = 1) in vec3 vnorm;
            layout(location = 2) in vec2 vtex;

            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;
            uniform float ofset;

            out vec2 TexCoords;
            out vec3 fragPos;
            out vec3 fragNorm;

            void main()
            {
                vec4 vertPos = model * vec4(vpos, 1.0);

                TexCoords = vtex + vec2(ofset, 0);
                fragPos = vertPos.xyz;
                fragNorm = (model * vec4(vnorm, 0)).xyz;
                gl_Position = projection * view * vertPos;
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;

            in vec2 TexCoords;
            in vec3 fragPos;
            in vec3 fragNorm;

            uniform sampler2D material;
            uniform vec3 camPos;
            uniform vec3 lightposition[1];
            uniform vec3 lightcolor[1];
            uniform float lightstrength[1];

            layout(location = 0) out vec4 color;

            vec3 calcPointlight(int i)
            {
                vec3 baseTexture = texture(material, TexCoords).rgb;
                vec3 result = vec3(0);

                vec3 relLightPos = lightposition[i] - fragPos;
                float distance = length(relLightPos);
                relLightPos = normalize(relLightPos);

                vec3 relCamPos = normalize(camPos - fragPos);
                vec3 halfVec = normalize(relLightPos + relCamPos);

                result += lightcolor[i] * lightstrength[i] * max(0.0, dot(fragNorm, relLightPos)) / (distance * distance) * baseTexture; //diffuse
                result += lightcolor[i] * lightstrength[i] * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / (distance * distance); //specular

                return result;
            }

            void main()
            {
                vec4 baseTex = texture(material, TexCoords);
                vec3 temp = 0.2 * baseTex.rgb; //ambient
                temp += calcPointlight(0);

                color = pow(vec4(temp, baseTex.a), vec4(0.45));
            }
        """,

        uniforms={'projection': projection.flatten(), 'view': np.identity(4).flatten(), 'model': np.identity(4).flatten(),
                  'camPos' : [0,0,0],
                  'lightposition': [[0, 1000, 0]],
                  'lightcolor': [[255,255,255]],
                  'lightstrength': [500],
                  'ofset': 0},

        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],

        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                         *zengl.bind(ctx.buffer(normBuffer), "3f", 1),
                         *zengl.bind(ctx.buffer(texBuffer), "2f", 2)],

        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )
def shader3Danimated(vertexBuffer, normBuffer, texBuffer, jointDataList, weightDataList, nrJoints, texture):

    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;

            layout(location = 0) in vec3 vpos;
            layout(location = 1) in vec3 vnorm;
            layout(location = 2) in vec2 vtex;
            layout(location = 3) in ivec4 vboneIds;
            layout(location = 4) in vec4 vweights;

            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;
            uniform mat4 animation[50];

            out vec2 TexCoords;
            out vec3 fragPos;
            out vec3 fragNorm;

            vec4 applyBone(vec4 p)
            {
                vec4 result = vec4(0.0);
                for(int i = 0; i < 4; ++i)
                {
                    if(vboneIds[i] >= 100)
                    {
                        result = p;
                        break;
                    }
                    result += vweights[i] * (animation[vboneIds[i]] * p);
                }
                return result;
            }

            void main()
            {

                vec4 position = applyBone(vec4(vpos, 1.0));
                vec4 normal = normalize(applyBone(vec4(vnorm, 0.0)));

                vec4 vertPos = model * position;

                TexCoords = vtex;
                fragPos = vertPos.xyz;
                fragNorm = (model * normal).xyz;
                gl_Position = projection * view * vertPos;
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;

            in vec2 TexCoords;
            in vec3 fragPos;
            in vec3 fragNorm;

            uniform sampler2D material;
            uniform vec3 camPos;
            uniform vec3 lightposition[1];
            uniform vec3 lightcolor[1];
            uniform float lightstrength[1];

            layout (location = 0) out vec4 color;

            vec3 calcPointlight(int i)
            {
                vec3 baseTexture = texture(material, TexCoords).rgb;
                vec3 result = vec3(0);

                vec3 relLightPos = lightposition[i] - fragPos;
                float distance = length(relLightPos);
                relLightPos = normalize(relLightPos);

                vec3 relCamPos = normalize(camPos - fragPos);
                vec3 halfVec = normalize(relLightPos + relCamPos);

                result += lightcolor[i] * lightstrength[i] * max(0.0, dot(fragNorm, relLightPos)) / (distance * distance) * baseTexture; //diffuse
                result += lightcolor[i] * lightstrength[i] * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / (distance * distance); //specular

                return result;
            }

            void main()
            {
                vec4 baseTex = texture(material, TexCoords);
                vec3 temp = 0.2 * baseTex.rgb; //ambient
                temp += calcPointlight(0);

                color = pow(vec4(temp, baseTex.a), vec4(0.45));
            }
        """,

        uniforms={'projection': projection.flatten(), 'view': np.identity(4).flatten(), 'model': np.identity(4).flatten(),
                  'animation': [np.identity(4) for i in range(nrJoints)],
                  'camPos': [0,0,0],
                  'lightposition': [[0, 0, 0]],
                  'lightcolor': [[255,255,255]],
                  'lightstrength': [100]},

        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 1}],
        resources=[{'type': 'sampler', 'binding': 1, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],

        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                         *zengl.bind(ctx.buffer(normBuffer), "3f", 1),
                         *zengl.bind(ctx.buffer(texBuffer), "2f", 2),
                         *zengl.bind(ctx.buffer(jointDataList), "4i", 3),
                         *zengl.bind(ctx.buffer(weightDataList), "4f", 4)],

        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )
def shader3Danitexnoise(vertexBuffer, normBuffer, texBuffer, texture, frameAmount):

    texBuffer /= [frameAmount, 1]

    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;

            layout(location = 0) in vec3 vpos;
            layout(location = 1) in vec3 vnorm;
            layout(location = 2) in vec2 vtex;

            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;
            uniform float ofset;

            out vec2 TexCoords;
            out vec3 fragPos;
            out vec3 fragNorm;

            void main()
            {
                vec4 vertPos = model * vec4(vpos, 1.0);

                TexCoords = vtex + vec2(ofset, 0);;
                fragPos = vertPos.xyz;
                fragNorm = (model * vec4(vnorm, 0)).xyz;
                gl_Position = projection * view * vertPos;
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;

            in vec2 TexCoords;
            in vec3 fragPos;
            in vec3 fragNorm;

            uniform sampler2D material;
            uniform vec3 camPos;
            uniform vec3 lightposition;
            uniform vec3 lightcolor;
            uniform float lightstrength;
            uniform float dis;

            layout(location = 0) out vec4 color;

            vec3 calcPointlight()
            {
                vec3 baseTexture = texture(material, TexCoords).rgb;
                vec3 result = vec3(0);

                vec3 relLightPos = lightposition - fragPos;
                float distance = length(relLightPos);
                relLightPos = normalize(relLightPos);

                vec3 relCamPos = normalize(camPos - fragPos);
                vec3 halfVec = normalize(relLightPos + relCamPos);

                result += lightcolor * lightstrength * max(0.0, dot(fragNorm, relLightPos)) / (distance * distance) * baseTexture; //diffuse
                result += lightcolor * lightstrength * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / (distance * distance); //specular

                return result;
            }

            void main()
            {
                vec4 baseTex = texture(material, TexCoords);
                vec3 temp = 0.2 * baseTex.rgb; //ambient
                temp += calcPointlight();
                temp += fract(sin(dot(TexCoords, vec2(12.9898, 78.233))) * 43758.5453) * dis;

                color = pow(vec4(temp, baseTex.a), vec4(0.45));
            }
        """,

        uniforms={'projection': projection.flatten(), 'view': np.identity(4).flatten(), 'model': np.identity(4).flatten(),
                  'camPos' : [0,0,0],
                  'lightposition': [0, 1000, 0],
                  'lightcolor': [255,255,255],
                  'lightstrength': 500,
                  'dis': 0,
                  'ofset': 0},

        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],

        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                         *zengl.bind(ctx.buffer(normBuffer), "3f", 1),
                         *zengl.bind(ctx.buffer(texBuffer), "2f", 2)],

        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )

#####################################################################################

class entity:

    def __init__(self, position, size, eulers= [0,0,0]):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.size = size

class player(entity):

    def __init__(self, position, eulers, camEulers, camZoom):

        super().__init__(position, 0, eulers)
        self.camera = camera(self.position, camEulers, camZoom)

    def update(self):

        cosX, sinX = self.camera.update(self.position)

        self.forwards = np.array((cosX, 0, -sinX))
        self.right = np.array((-sinX, 0, -cosX))

    def angle(self, frameTime, dPos):

        angle = np.arctan2(dPos[1], -dPos[0])

        angle += self.camera.eulers[0] + self.eulers[2]
        angle %= 2*np.pi
        if angle > np.pi:
            self.eulers[2] += (2*np.pi-angle)* frameTime * 0.01
        else:
            self.eulers[2] -= angle* frameTime * 0.01
        self.eulers[2] %= 2*np.pi

    def move(self, movement):

        self.position += movement

class camera(entity):

    def __init__(self, position, eulers, camZoom):

        super().__init__(position + (0,0.15,0), 0, eulers)
        self.zoom = camZoom
        self.update(position)

    def update(self, pos):

        angleX = self.eulers[0]
        angleY = self.eulers[1]

        cosX = np.cos(angleX)
        sinX = np.sin(angleX)
        cosY = np.cos(angleY)
        sinY = np.sin(angleY)

        self.forwards = np.array((cosX*cosY, sinY, -sinX*cosY))
        self.playerForwards = (cosX, 0, -sinX)
        self.right = (sinX, 0, cosX)
        self.up = np.array((-cosX*sinY, cosY, sinX*sinY))

        self.position = pos + (0,0.15,0) - self.zoom * self.forwards

        self.makeFrustum()

        return (cosX, sinX)

    def getViewTransform(self):

        return np.array(((self.right[0], self.up[0], -self.forwards[0], 0),
                         (self.right[1], self.up[1], -self.forwards[1], 0),
                         (self.right[2], self.up[2], -self.forwards[2], 0),
                         (-np.dot(self.right, self.position), -np.dot(self.up, self.position), np.dot(self.forwards,self.position), 1.0)), dtype=np.float32)

    def getYawMat(self):

        return np.array(((self.right[0], 0, -self.playerForwards[0], 0),
                         (self.right[1], 1, -self.playerForwards[1], 0),
                         (self.right[2], 0, -self.playerForwards[2], 0),
                         (-np.dot(self.right, self.position), -np.dot((0,1,0), self.position), np.dot(self.playerForwards,self.position), 1.0)), dtype=np.float32)

    def spin(self, dEulers):

        self.eulers += dEulers

        self.eulers[0] %= 2*np.pi
        self.eulers[1] = min(1.5, max(-1.5, self.eulers[1]))

    def makeFrustum(self):

        self.frustumParts = [normalize(self.forwards + self.right), normalize(self.forwards - self.right), normalize(self.forwards + self.up * 16/9), normalize(self.forwards - self.up * 16/9)]
        self.frustum = [np.append(normal, np.sum(normal * self.position)) for normal in self.frustumParts]

class scene:

    def __init__(self, sceneNr, playerPos, playerEul, camEul, camZoom):

        global mainMenuMusic
        self.minigameDone = False
        self.startTime = None
        self.player = player(playerPos, playerEul, camEul, camZoom)
        self.jumpTime = 0
        self.height = 0
        lights = [[0, 3, 0], [1,1,1], 1]

        vertices = np.array([[-1,-1], [1,-1], [1,1], [-1,-1], [1,1], [-1,1]], dtype=np.float32)
        texCoords = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)

        if sceneNr in [SCENES["anim0"], SCENES["anim1"], SCENES["beforeGame1"], SCENES["afterGame1"], SCENES["beforeGame2"], SCENES["afterGame2"], SCENES["beforeGame3"], SCENES["afterGame3"], SCENES['credits']]:

            if sceneNr == SCENES["anim0"]:

                mixer.fadeout(1000)

                SFXchannel.play(crowdNoise)
                SFXchannel.fadeout(8000)

                self.sound = introDialogue
                self.animation = shader2Danitex(vertices, texCoords, material("gfx/animations/anim0.png").img, 3)
                self.frames = 3
                self.scenelenght = 22

            if sceneNr == SCENES["anim1"]:

                mixer.fadeout(1000)

                self.sound = POGsong
                self.animation = shader2Danitex(vertices, texCoords, material("gfx/animations/anim1.png").img, 4)
                self.frames = 4
                self.scenelenght = 210

            if sceneNr == SCENES["beforeGame1"]:

                mixer.fadeout(1000)
                musicChannel.play(paranoidMusic, -1, 0, 1000)

                self.sound = game1Dialogue2
                self.animation = shader2D(vertices, texCoords, material("gfx/animations/beforeGame1.png").img)
                self.scenelenght = 8

            if sceneNr == SCENES['afterGame1']:

                musicChannel.fadeout(1000)
                musicChannel.play(hopefulMusic, -1, 0, 1000)

                self.sound = game1Dialogue3
                self.animation = shader2D(vertices, texCoords, material("gfx/animations/afterGame1.png").img)
                self.scenelenght = 5

            if sceneNr == SCENES['beforeGame2']:

                musicChannel.fadeout(1000)
                musicChannel.play(paranoidMusic, -1, 0, 1000)

                self.sound = game2Dialogue2
                self.animation = shader2D(vertices, texCoords, material("gfx/animations/beforeGame2.png").img)
                self.scenelenght = 13

            if sceneNr == SCENES['afterGame2']:

                musicChannel.fadeout(1000)
                musicChannel.play(hopefulMusic, -1, 0, 1000)

                self.sound = game2Dialogue3
                self.animation = shader2D(vertices, texCoords, material("gfx/animations/afterGame2.png").img)
                self.scenelenght = 24

            if sceneNr == SCENES['beforeGame3']:

                musicChannel.fadeout(1000)
                musicChannel.play(paranoidMusic, -1, 0, 1000)

                self.sound = game3Dialogue1
                self.animation = shader2D(vertices, texCoords, material("gfx/animations/beforeGame3.png").img)
                self.scenelenght = 35

            if sceneNr == SCENES['afterGame3']:

                musicChannel.fadeout(1000)
                musicChannel.play(hopefulMusic, -1, 0, 1000)

                self.sound = game3Dialogue2
                self.animation = shader2D(vertices, texCoords, material("gfx/animations/afterGame3.png").img)
                self.scenelenght = 12

            if sceneNr == SCENES['credits']:

                if not mainMenuMusic:
                    mainMenuMusic = True
                    musicChannel.fadeout(1000)
                    musicChannel.play(titlescreenMusic, -1, 0, 1000)

                self.sound = None
                self.animation = shader2D(vertices, texCoords, material("gfx/Credits.png").img)
                self.scenelenght = 20

            self.entities = {}
            self.interact = None
            self.minigame = None

            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)

        elif sceneNr == SCENES['vedals room1']:

            musicChannel.fadeout(1000)
            musicChannel.play(pianoMusic, -1, 0, 1000)
            self.player
            self.entities = {
                ENTITY_TYPE["player"]: gameObjects[ENTITY_TYPE["player"]],
                ENTITY_TYPE['keyUI']:  gameObjects[ENTITY_TYPE['keyUI']],
                ENTITY_TYPE["room"]:   gameObjects[ENTITY_TYPE["room"]],
                ENTITY_TYPE["bed"]:    gameObjects[ENTITY_TYPE["bed"]],
                ENTITY_TYPE["laptop"]: gameObjects[ENTITY_TYPE["laptop"]],
                ENTITY_TYPE["stool"]:  gameObjects[ENTITY_TYPE["stool"]],
                ENTITY_TYPE["desk"]:   gameObjects[ENTITY_TYPE["desk"]],
                ENTITY_TYPE["chair"]:  gameObjects[ENTITY_TYPE["chair"]],
                ENTITY_TYPE["tablet"]: gameObjects[ENTITY_TYPE["tablet"]]
                }
            self.entities[ENTITY_TYPE["player"]][0] = self.player

            self.interact = ENTITY_TYPE["chair"]
            self.minigame = None

            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)

        elif sceneNr == SCENES['minigame1']:

            self.entities = {
                ENTITY_TYPE["player"]: gameObjects[ENTITY_TYPE["player"]],
                ENTITY_TYPE["room"]:   gameObjects[ENTITY_TYPE["room"]],
                ENTITY_TYPE["desk"]:   gameObjects[ENTITY_TYPE["desk"]],
                ENTITY_TYPE["chair"]:  gameObjects[ENTITY_TYPE["chair"]]
                }
            self.entities[ENTITY_TYPE["player"]][0] = self.player

            self.interact = None
            self.minigame = filterMinigame()

            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)

        elif sceneNr == SCENES['vedals room2']:

            musicChannel.play(musicboxMusic, -1, 0, 1000)

            self.entities = {
                ENTITY_TYPE["player"]: gameObjects[ENTITY_TYPE["player"]],
                ENTITY_TYPE['keyUI']:  gameObjects[ENTITY_TYPE['keyUI']],
                ENTITY_TYPE["room"]:   gameObjects[ENTITY_TYPE["room"]],
                ENTITY_TYPE["bed"]:    gameObjects[ENTITY_TYPE["bed"]],
                ENTITY_TYPE["laptop"]: gameObjects[ENTITY_TYPE["laptop"]],
                ENTITY_TYPE["stool"]:  gameObjects[ENTITY_TYPE["stool"]],
                ENTITY_TYPE["desk"]:   gameObjects[ENTITY_TYPE["desk"]],
                ENTITY_TYPE["chair"]:  gameObjects[ENTITY_TYPE["chair"]],
                ENTITY_TYPE["tablet"]: gameObjects[ENTITY_TYPE["tablet"]]
                }
            self.entities[ENTITY_TYPE["player"]][0] = self.player

            self.interact = ENTITY_TYPE["laptop"]
            self.minigame = audioMinigame()

            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)

        elif sceneNr == SCENES['minigame2']:

            self.entities = {
                ENTITY_TYPE["player"]: gameObjects[ENTITY_TYPE["player"]],
                ENTITY_TYPE["room"]:   gameObjects[ENTITY_TYPE["room"]],
                ENTITY_TYPE["bed"]:    gameObjects[ENTITY_TYPE["bed"]],
                ENTITY_TYPE["laptop"]: gameObjects[ENTITY_TYPE["laptop"]],
                ENTITY_TYPE["stool"]:  gameObjects[ENTITY_TYPE["stool"]],
                ENTITY_TYPE["tablet"]: gameObjects[ENTITY_TYPE["tablet"]]
                }
            self.entities[ENTITY_TYPE["player"]][0] = self.player

            self.interact = None
            self.minigame = audioMinigame()

            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

        elif sceneNr == SCENES['vedals room3']:

            musicChannel.play(pianoMusic, -1, 0, 1000)

            self.entities = {
                ENTITY_TYPE["player"]: gameObjects[ENTITY_TYPE["player"]],
                ENTITY_TYPE['keyUI']:  gameObjects[ENTITY_TYPE['keyUI']],
                ENTITY_TYPE["room"]:   gameObjects[ENTITY_TYPE["room"]],
                ENTITY_TYPE["bed"]:    gameObjects[ENTITY_TYPE["bed"]],
                ENTITY_TYPE["laptop"]: gameObjects[ENTITY_TYPE["laptop"]],
                ENTITY_TYPE["stool"]:  gameObjects[ENTITY_TYPE["stool"]],
                ENTITY_TYPE["desk"]:   gameObjects[ENTITY_TYPE["desk"]],
                ENTITY_TYPE["chair"]:  gameObjects[ENTITY_TYPE["chair"]],
                ENTITY_TYPE["tablet"]: gameObjects[ENTITY_TYPE["tablet"]]
                }
            self.entities[ENTITY_TYPE["player"]][0] = self.player

            self.interact = ENTITY_TYPE["stool"]
            self.minigame = None

            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)

        elif sceneNr == SCENES['minigame3']:

            self.entities = {
                ENTITY_TYPE["player"]: gameObjects[ENTITY_TYPE["player"]],
                ENTITY_TYPE['keyUI']:  gameObjects[ENTITY_TYPE['keyUI']],
                ENTITY_TYPE["room"]:   gameObjects[ENTITY_TYPE["room"]],
                ENTITY_TYPE["stool"]:  gameObjects[ENTITY_TYPE["stool"]],
                ENTITY_TYPE["desk"]:   gameObjects[ENTITY_TYPE["desk"]],
                ENTITY_TYPE["chair"]:  gameObjects[ENTITY_TYPE["chair"]],
                ENTITY_TYPE["tablet"]: gameObjects[ENTITY_TYPE["tablet"]]
                }
            self.entities[ENTITY_TYPE["player"]][0] = self.player

            self.interact = None
            self.minigame = videoMinigame()

            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

        self.entityGrid = [[[] for j in range(500)] for i in range(500)]
        for entity_type, obj in self.entities.items():

            #set lights
            for shader in obj[1].shaders:

                shader.uniforms['lightposition'][:] = struct.pack('3f', *lights[0])
                shader.uniforms['lightcolor'][:] = struct.pack('3f', *lights[1])
                shader.uniforms['lightstrength'][:] = struct.pack('1f', lights[2])

            #skip non-collision objects
            if entity_type in [ENTITY_TYPE["player"], ENTITY_TYPE["keyUI"], ENTITY_TYPE["bounding_box"]]: continue

            meshBoundingBoxes = obj[1].boundingBox + obj[0].position
            for meshBoundingBox in meshBoundingBoxes:

                meshmin, meshmax = [(int(vec3[0]), int(vec3[2])) for vec3 in meshBoundingBox/4 + 250]
                [self.entityGrid[x][y].append(obj) for x in range(meshmin[0]-1, meshmax[0]+2) for y in range(meshmin[1]-1, meshmax[1]+2) if obj not in self.entityGrid[x][y]]

    def jump(self, jump):

        if not self.jumpTime:
            self.jumpTime = jump
            self.jumpStartHeight = self.height

        t = (jump - self.jumpTime)
        jumpheight = 3*t - 4.9*(t**2)

        if jumpheight < (self.height - self.jumpStartHeight):
            self.player.position[1] = self.height
            self.jumpTime = 0
            return False
        else:
            self.player.position[1] = self.jumpStartHeight + jumpheight
            return True

    def movePlayer(self, dPos, sprint, frametime):

        movement = normalize((dPos[0]*self.player.right + dPos[1]*self.player.forwards)) * (1 + sprint)
        movement, collisionHeight = self.checkCollision(movement, self.player.position)

        self.player.angle(frametime, dPos)
        self.player.move(movement * 0.002 * frametime)

        #pos = self.player.position[0:3:2] * 5/2 + 2500
        #pos = [int(i) for i in pos]

        #mapHeight = [self.heightmap.pixels.getpixel((pos[1] + x, pos[0] + y))[1]/32 for x, y in [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]]
        #angle = [np.arctan(mapHeight[x] - mapHeight[y]) for x, y in [(0, 2), (2, 4), (2, 1), (3, 2)]]

        #roll = (angle[0] + angle[1]) * 0.5
        #pitch = (angle[2] + angle[3]) * 0.5

        #self.player.eulers[0] = pitch
        #self.player.eulers[1] = roll

        #self.height = max(mapHeight[2] * 31.25/64, collisionHeight) - 0.1
        self.height = collisionHeight - 0.1

        if not self.jumpTime:
            self.player.position[1] = self.height

    def checkCollision(self, movement, pos):

        meshBoundingBoxList, collisionPosList, movementList, distanceList, heightList = [], [], [], [], []
        collisionHeight = 0

        cell = pos[::2]/4 + 250
        grid = self.entityGrid[int(cell[0])][int(cell[1])]
        for obj in grid:

            meshBoundingBoxes = obj[1].boundingBox + obj[0].position
            for meshBoundingBox in meshBoundingBoxes:

                localBoundingBox = meshBoundingBox - pos

                if all(localBoundingBox[0][2*i] < 0 < localBoundingBox[1][2*i] for i in range(2)) and localBoundingBox[1][1] < 1:
                    heightList.append(localBoundingBox[1][1])

                if localBoundingBox[1][1] < 0.4: continue

                moveRay = np.array(((0,0,0), movement), dtype=np.float32)
                collisionPos = ray_intersect_aabb(moveRay, localBoundingBox)
                if collisionPos is not None:

                    distance = np.linalg.norm(collisionPos)
                    if distance < 1:

                        normal = self.getNormal(collisionPos, localBoundingBox)
                        leftoverMovement = movement - collisionPos
                        leftoverMovement -= (normal @ leftoverMovement) * normal

                        distanceList.append(distance)
                        collisionPosList.append(collisionPos)
                        movementList.append(leftoverMovement)
                        meshBoundingBoxList.append(meshBoundingBox)

        if distanceList:
            index = distanceList.index(min(distanceList))
            movement = collisionPosList[index]
            movement += self.checkCollision2(movementList[index], meshBoundingBoxList-pos+movement)

        if heightList:
            collisionHeight = pos[1] + max(heightList)

        return movement, collisionHeight

    def checkCollision2(self, movement, meshBoundingBoxList):

        collisionPosList, distanceList = [], []

        for meshBoundingBox in meshBoundingBoxList:

            moveRay = np.array(((0,0,0), movement), dtype=np.float32)
            collisionPos = ray_intersect_aabb(moveRay, meshBoundingBox)
            if collisionPos is not None:

                distanceList.append(np.linalg.norm(collisionPos))
                collisionPosList.append(collisionPos)

        if distanceList:
            index = distanceList.index(min(distanceList))
            movement = collisionPosList[index]

        return movement

    def getNormal(self, collisionPos, boundingBox):

        distances = [abs(collisionPos[0] - boundingBox[0][0]),
                     abs(collisionPos[0] - boundingBox[1][0]),
                     abs(collisionPos[2] - boundingBox[0][2]),
                     abs(collisionPos[2] - boundingBox[1][2])]

        index = distances.index(min(distances))

        if index == 0: return np.array((1,0,0))
        if index == 1: return np.array((-1,0,0))
        if index == 2: return np.array((0,0,1))
        if index == 3: return np.array((0,0,-1))

    def moveCam(self, dEulers):

        self.player.camera.spin(dEulers)
        playpos = self.player.position + (0,0.15,0)

        ofset = normalize(self.player.camera.position - playpos)

        zoom = self.camCollision(ofset, playpos)
        self.player.camera.zoom = min(zoom, 2.5)

    def camCollision(self, ofset, pos):

        collisionPosList, distanceList = [], []

        cell = pos[::2]/4 + 250
        grid = self.entityGrid[int(cell[0])][int(cell[1])]

        for obj in grid:

            meshBoundingBoxes = obj[1].boundingBox + obj[0].position
            for meshBoundingBox in meshBoundingBoxes:

                localBoundingBox = meshBoundingBox - pos

                moveRay = np.array(((0,0,0), ofset), dtype=np.float32)
                collisionPos = ray_intersect_aabb(moveRay, localBoundingBox)
                if collisionPos is not None:

                    normal = self.getNormal(collisionPos, localBoundingBox)
                    leftoverOfset = ofset - collisionPos
                    leftoverOfset -= (normal @ leftoverOfset) * normal

                    distance = np.linalg.norm(collisionPos)
                    distanceList.append(distance)
                    collisionPosList.append(collisionPos)

        if distanceList:
            index = distanceList.index(min(distanceList))
            ofset = collisionPosList[index]

        return np.linalg.norm(ofset)

    def action(self, sceneNr, keyInteract, arrowL, arrowR, relMousePos, click, drag, time):

        if sceneNr in [SCENES["anim0"], SCENES["anim1"]]:

            if self.startTime == None:
                self.startTime = time
                self.scroll = 0
                dialogueChannel.play(self.sound)
            if time - self.startTime > self.scenelenght:
                return 1

            if time - self.startTime > 2:
                self.scroll = (time - self.startTime) - 2

            self.animation.uniforms['ofset'][:] = struct.pack('1f', min((self.scroll/6)/self.frames, (self.frames-1)/self.frames))
            self.animation.render()

        elif sceneNr in [SCENES["credits"], SCENES["beforeGame1"], SCENES["afterGame1"], SCENES["beforeGame2"], SCENES["afterGame2"], SCENES["beforeGame3"], SCENES["afterGame3"]]:

            if self.startTime == None:
                self.startTime = time
                if self.sound:
                    SFXchannel.play(self.sound)

            if time - self.startTime > self.scenelenght:
                return 1

            self.animation.render()

        elif sceneNr in [SCENES["vedals room1"], SCENES["vedals room2"], SCENES["vedals room3"]]:

            interactPos = self.entities[self.interact][0].position
            relPos = interactPos - self.player.position

            eulers = self.player.camera.eulers

            self.entities[ENTITY_TYPE['keyUI']][0].position = interactPos + [0,0.5,0] - 0.5 * normalize(np.array([relPos[0], 0, relPos[2]]))
            self.entities[ENTITY_TYPE['keyUI']][0].eulers[2] = -eulers[0]

            if keyInteract:
                return 1

            return 0

        elif sceneNr == SCENES['minigame1']:

            if self.startTime and (time - self.startTime > 1):
                return 2

            if any([arrowL, arrowR]):
                state = self.minigame.checkword(arrowR)
                if state == 2:
                    self.startTime = time
                    return 0
                else:
                    return state

        elif sceneNr == SCENES['minigame2']:

            if self.minigame.handleMouse(pygame.mouse.get_pos(), drag, relMousePos, click):
                return 1
            return 0

        elif sceneNr == SCENES['minigame3']:

            if self.minigame.handleMouse(pygame.mouse.get_pos(), click or drag, relMousePos, time):
                return 1

            return 0

    def render(self, frametime):

        cam = self.player.camera
        view = cam.getViewTransform()
        frustum = cam.frustum

        for entity in self.entities.values():
            obj, mesh = entity

            if all([vec4[0:3] @ obj.position - vec4[3] > -entity[0].size for vec4 in frustum]):

                transformMat = np.identity(4)
                transformMat[0:3,0:3] = create_from_eulers(entity[0].eulers)
                transformMat[3,0:3] = obj.position

                if mesh.hasJoints:
                    mesh.pose += frametime
                    mesh.setUniform()

                mesh.draw(view, transformMat, cam.position)

        if self.minigame:
            self.minigame.draw(view, cam.position)

        image.blit(output)
        output.blit()
        ctx.end_frame()
        pygame.display.flip()

class game:

    def __init__(self, sceneNr= None):

        global mainMenuMusic

        self.drag = False
        self.clicktime = 0
        self.jump = 0

        if sceneNr:
            self.sceneNr = sceneNr
        else:
            self.sceneNr = SCENES["anim0"]
            mainMenuMusic = False

        self.scene = scene(self.sceneNr, [0,0,0], [0,0,0], [0,0,0], 3)
        self.set_up_timer()
        self.scenestartTime = self.time
        self.gameLoop()

    def gameLoop(self):

        result = CONTINUE
        arrowL, arrowR, click = False, False, False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result = EXIT
            elif event.type == pygame.KEYDOWN:
                if event.key == input_map["escape"]:
                    result = OPEN_MENU
                elif event.key == input_map['leftArrow']:
                    arrowL = True
                elif event.key == input_map['rightArrow']:
                    arrowR = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.drag = True
                self.clicktime = self.time
            elif event.type == pygame.MOUSEBUTTONUP:
                if self.time - self.clicktime <0.5:
                    click = True
                self.drag = False

        self.calculate_framerate()
        relpos = self.handle_mouse()
        interact = self.handle_keys()

        actionResult = self.scene.action(self.sceneNr, interact, arrowL, arrowR, relpos, click, self.drag, self.time)
        if actionResult:
            self.jump = False
            if actionResult == 1:
                self.sceneNr += 1
            if self.sceneNr > SCENES["credits"]:
                return OPEN_MENU

            elif self.sceneNr in [SCENES["anim0"], SCENES["anim1"], SCENES["beforeGame1"], SCENES["afterGame1"], SCENES["beforeGame2"], SCENES["afterGame2"], SCENES["beforeGame3"], SCENES["afterGame3"], SCENES["credits"]]:
                self.scenestartTime = self.time
                self.scene = scene(self.sceneNr, [0,0,0], [0,0,0], [0,0,0], 0)

            elif self.sceneNr in [SCENES["vedals room1"], SCENES["vedals room2"], SCENES["vedals room3"]]:
                self.scenestartTime = self.time
                self.scene = scene(self.sceneNr, [0,0,-1], [0,0,0], [-np.pi/2,-np.pi/8,0], 2)

            elif self.sceneNr == SCENES["minigame1"]:
                self.scenestartTime = self.time
                self.scene = scene(self.sceneNr, [0.51,1.2,1], [0,0,-np.pi/2], [-np.pi/30,-np.pi/8,0], 0)
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)

            elif self.sceneNr == SCENES["minigame2"]:
                self.scenestartTime = self.time
                self.scene = scene(self.sceneNr, [-1.5,0.4,1.2], [0,0,-np.pi/2], [np.pi/2,-np.pi/6,0], 0)

            elif self.sceneNr == SCENES["minigame3"]:
                self.scenestartTime = self.time
                self.scene = scene(self.sceneNr, [1.11,0.57,-1.21], [0,0,-np.pi/2+np.pi*0.35], [-np.pi*0.35,-np.pi/3,0], 0)
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)

        self.scene.player.update()
        self.scene.render(self.frametime)

        return result

    def handle_keys(self):

        dPos = 0
        keys = pygame.key.get_pressed()
        interact = False

        if self.sceneNr in [SCENES["vedals room1"], SCENES["vedals room2"], SCENES["vedals room3"]]:
            if keys[input_map["forwards"]]:
                dPos += np.array([0,1])
            if keys[input_map["left"]]:
                dPos += np.array([1,0])
            if keys[input_map["backwards"]]:
                dPos -= np.array([0,1])
            if keys[input_map["right"]]:
                dPos -= np.array([1,0])
            if keys[input_map["jump"]]:
                self.jump = True

        #this could have been an if statement but this is cooler
        sprint = keys[input_map["sprint"]]

        #the jump code is an ungodly mess, dont touch it if not needed
        if self.jump:
            self.jump = self.scene.jump(self.time)

        if np.any(dPos):
            self.scene.movePlayer(dPos, sprint, self.frametime)

        if keys[input_map["interact"]]:
            interact = True

        return interact

    def handle_mouse(self):

        (x,y) = pygame.mouse.get_rel()
        if self.sceneNr in [SCENES["vedals room1"], SCENES["vedals room2"], SCENES["vedals room3"]]:
            dEulers = -0.001 * np.array([x,y,0])
            self.scene.moveCam(dEulers)
        return x,y

    def set_up_timer(self):

        self.last_time = 0
        self.time = 0
        self.frametime = 0

    def calculate_framerate(self):

        clock.tick(FPS)
        framerate = clock.get_fps()
        if framerate != 0:
            self.frametime = 1000/framerate

        self.time = pygame.time.get_ticks()/1000
        if self.time - self.last_time > 1:
            pygame.display.set_caption(f"Running at {int(framerate)} fps.")
            self.last_time = self.time

class filterMinigame:

    def __init__(self):

        self.wordDict = {0: ["ducky", True],
                         1: ["frogs", True],
                         2: ["evil", True],
                         3: ["vedal", True],
                         4: ["eepy", True],
                         5: ["Neuro", True],
                         6: ["woof", True],
                         7: ["meow", True],
                         8: ["corpa", True],
                         9: ["cookies", True],
                         10: ["classic", True],
                         11: ["welcome", True],
                         12: ["timeout", True],
                         13: ["demons", True],
                         14: ["British", True],
                         15: ["clap", True],
                         16: ["gymbag", True],
                         17: ["Based", True],
                         18: ["Life", True],
                         19: ["misinformation", True],
                         20: ["source-code.py", False],
                         21: ["backslash", False],
                         22: ["hiyori", False],
                         23: ["kill me", False],
                         24: ["uwu", False],
                         25: ["raw fucky", False],}

        vertices1 = np.array([[1.910,1.002,0.8783], [1.910,1.002,1.502], [1.910,1.314,1.502], [1.910,1.002,0.8783], [1.910,1.314,1.502], [1.910,1.314,0.8783]], dtype=np.float32)
        vertices2 = np.array([[1.444,1.002,0.4054], [1.893,1.002,0.8385], [1.893,1.314,0.8385], [1.444,1.002,0.4054], [1.893,1.314,0.8385], [1.444,1.314,0.4054]], dtype=np.float32)
        vertices3 = np.array([[1.893,1.002,1.542], [1.444,1.002,1.975], [1.444,1.314,1.975], [1.893,1.002,1.542], [1.444,1.314,1.975], [1.893,1.314,1.542]], dtype=np.float32)
        normals = np.array([[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]], dtype=np.float32)
        texCoords = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)
        texture1 = material("gfx/3D_model_textures/words.png")
        texture2 = material("gfx/3D_model_textures/keyLeft.png")
        texture3 = material("gfx/3D_model_textures/keyRight.png")
        brokenScreen = material("gfx/brokenScreen.jpg")

        self.shaders = [[shader3D(vertices2, normals, texCoords, brokenScreen.img),
                         shader3D(vertices3, normals, texCoords, brokenScreen.img),
                         shader3D(vertices1, normals, texCoords, brokenScreen.img)],
                        [shader3D(vertices2, normals, texCoords, texture2.img),
                         shader3D(vertices3, normals, texCoords, texture3.img),
                         shader3Danitex(vertices1, normals, texCoords, texture1.img, len(self.wordDict)),]]

        vertices4 = np.array([[-1,-1], [1,-1], [1,1], [-1,-1], [1,1], [-1,1]], dtype=np.float32)
        texCoords2 = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)
        filter = material("gfx/animations/filterJumpscare.png")

        self.filter = shader2D(vertices4, texCoords2, filter.img)

        self.list = np.arange(26)
        self.state = 0
        np.random.shuffle(self.list)

        self.shaders[1][2].uniforms['ofset'][:] = struct.pack("1f", self.list[0] / len(self.wordDict))

    def checkword(self, arrowR):

        if len(self.list) == 0:
            return 1

        index = self.list[0]
        word = self.wordDict[index]

        if word[1] == arrowR:
            SFXchannel.play(wrongBuzzer)
            self.list = np.append(self.list, index)
            self.state += 1
            if self.state == 3:
                return 2
        else:
            SFXchannel.play(trueBuzzer)

        self.list = self.list[1:]
        if len(self.list) == 0:
            return 1

        self.shaders[1][2].uniforms['ofset'][:] = struct.pack("1f", self.list[0] / len(self.wordDict))
        return 0

    def draw(self, view, camPos):

        drawList = [self.shaders[1 - min(max(0, self.state - j), 1)][j] for j in range(3)]

        for shader in drawList:
            shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
            shader.uniforms['camPos'][:] = struct.pack('3f', *camPos)
            shader.render()

        if self.state == 3:
            self.filter.render()

class audioMinigame:

    def __init__(self):

        self.sound = mixer.Sound("sfx/minigame2/audio3.ogg")
        self.lastclick = False
        ofset = np.random.rand(10, 2)

        self.soundDict = {0: dragPlane(ofset[0], True, mixer.Sound("sfx/minigame2/audio3.ogg")),
                          1: dragPlane(ofset[1], True, mixer.Sound("sfx/minigame2/audio5.ogg")),
                          2: dragPlane(ofset[2], True, mixer.Sound("sfx/minigame2/audio7.ogg")),
                          3: dragPlane(ofset[3], True, mixer.Sound("sfx/minigame2/audio8.ogg")),
                          4: dragPlane(ofset[4], False, mixer.Sound("sfx/minigame2/verygarbled1.ogg")),
                          5: dragPlane(ofset[5], False, mixer.Sound("sfx/minigame2/verygarbled2.ogg")),
                          6: dragPlane(ofset[6], False, mixer.Sound("sfx/minigame2/verygarbled4.ogg")),
                          7: dragPlane(ofset[7], False, mixer.Sound("sfx/minigame2/verygarbled6.ogg")),
                          8: dragPlane(ofset[8], False, mixer.Sound("sfx/minigame2/verygarbled9.ogg")),
                          9: dragPlane(ofset[9], False, mixer.Sound("sfx/minigame2/verygarbled10.ogg"))}

        vertices = np.array([[-1.595,0.4226,0.988], [-1.405,0.4226,0.988], [-1.405,0.5323,0.948], [-1.595,0.4226,0.988], [-1.405,0.5323,0.948], [-1.595,0.5323,0.948]], dtype=np.float32)
        normals = np.array([[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1]], dtype=np.float32)
        texCoords = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)
        texture = material("gfx/3D_model_textures/laptopBG.png")

        self.shader = shader3D(vertices, normals, texCoords, texture.img)

    def handleMouse(self, pos, drag, relpos, click):

        remove = None

        for index, plane in self.soundDict.items():
            sound = plane.handleMouse(pos, drag, np.array(relpos), click)
            if sound == 1:
                if plane.value:
                    SFXchannel.play(trueBuzzer)
                    remove = index + 1
                    plane.drag = False
                else:
                    SFXchannel.play(wrongBuzzer)
                    ofset = np.random.rand(2)
                    plane.setLocation(ofset)
                    plane.drag = False
            elif sound == 2:
                if not plane.value:
                    SFXchannel.play(trueBuzzer)
                    remove = index + 1
                    plane.drag = False
                else:
                    SFXchannel.play(wrongBuzzer)
                    ofset = np.random.rand(2)
                    plane.setLocation(ofset)
                    plane.drag = False
            elif sound:
                SFXchannel.play(sound)

        if remove:
            del self.soundDict[remove - 1]

        if len(self.soundDict) == 0:
            return 1
        return 0

    def draw(self, view, camPos):

        self.shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
        self.shader.uniforms['camPos'][:] = struct.pack('3f', *camPos)
        self.shader.render()

        for plane in self.soundDict.values():

            plane.draw(view, camPos)

class videoMinigame:

    def __init__(self):

        vertices = np.array([[1.188,0.596,-1.203], [1.071,0.596,-1.141], [1.112,0.596,-1.064], [1.188,0.596,-1.203], [1.112,0.596,-1.064], [1.229,0.596,-1.126]], dtype=np.float32)
        normals = np.array([[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]], dtype=np.float32)
        texCoords = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)
        texture = material("gfx/emotes/emotes.png")
        self.frameCount = 12
        self.emoteNr = 0
        self.time = None

        self.shader = shader3Danitexnoise(vertices, normals, texCoords, texture.img, 12)
        sliderPos = 0.5
        self.sliderGoal = np.random.rand()
        self.shader.uniforms['dis'][:] = struct.pack('1f', self.sliderGoal - sliderPos)

        self.slider = slider(sliderPos)

    def handleMouse(self, pos, click, relpos, time):

        ofset = self.slider.handleMouse(pos, click, relpos)
        self.shader.uniforms['dis'][:] = struct.pack('1f', self.sliderGoal - ofset)
        if abs(self.sliderGoal - ofset) < 0.04:
            if self.time is None:
                self.time = time
            elif time - self.time > 1:
                self.time = None
                self.emoteNr += 1

                if self.emoteNr == self.frameCount:
                    return 1

                SFXchannel.play(trueBuzzer)
                self.sliderGoal = np.random.rand()
                self.shader.uniforms['ofset'][:] = struct.pack('1f', self.emoteNr / self.frameCount)
        else:
            self.time = None
        return 0

    def draw(self, view, camPos):

        self.shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
        self.shader.uniforms['camPos'][:] = struct.pack('3f', *camPos)
        self.shader.render()
        self.slider.draw(view, camPos)

class menu:

    def __init__(self):

        global mainMenuMusic
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

        if not mainMenuMusic:
            mixer.stop()
            mainMenuMusic = True
            musicChannel.play(titlescreenMusic, -1, 0, 1000)

        self.createObjects()
        self.set_up_timer()
        self.gameLoop()

    def createObjects(self):

        vertices = np.array([[-1,-1], [1,-1], [1,1], [-1,-1], [1,1], [-1,1]], dtype=np.float32)
        texCoords = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)
        BG = material("gfx/menu.png")
        self.BG = shader2D(vertices, texCoords, BG.img)

        self.buttons = []

        newGameButton = button((0, 0.05), (0.7, 0.2), material("gfx/start.png"), NEW_GAME)
        self.buttons.append(newGameButton)

        quitButton = button((0, -0.2), (0.7, 0.2), material("gfx/creditsbutton.png"), CREDITS)
        self.buttons.append(quitButton)

    def gameLoop(self):

        result = CONTINUE
        click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result = EXIT
            elif event.type == pygame.MOUSEBUTTONDOWN:
                click = True

        self.calculate_framerate()

        if result == CONTINUE:
            result = self.handleMouse(click)

        ctx.new_frame()
        image.clear()
        depth.clear()

        for button in self.buttons:
            button.shader.render()

        self.BG.render()

        image.blit(output)
        output.blit()
        ctx.end_frame()

        pygame.display.flip()

        return result

    def handleMouse(self, click):

        (x,y) = pygame.mouse.get_pos()
        x -= WIDTH * 0.5
        x /= WIDTH * 0.5
        y -= HEIGHT * 0.5
        y /= -HEIGHT * 0.5

        for button in self.buttons:
            result = button.handleMouse((x,y), click)
            if result != CONTINUE:
                return result
        return CONTINUE

    def set_up_timer(self):

        self.last_time = pygame.time.get_ticks()/1000
        self.frametime = 0

    def calculate_framerate(self):

        clock.tick(FPS)
        framerate = clock.get_fps()
        if framerate != 0:
            self.frametime = 1000/framerate

        time = pygame.time.get_ticks()/1000
        if time - self.last_time > 1:
            pygame.display.set_caption(f"Running at {int(framerate)} fps.")
            self.last_time = time

#####################################################################################

class button:

    def __init__(self, pos, size, texture, function):

        self.click = function
        self.pos = pos
        self.size = size
        self.frameCount = 2

        vertices = np.array([[pos[0] - self.size[0]*0.5, pos[1] - self.size[1]*0.5],
                             [pos[0] + self.size[0]*0.5, pos[1] - self.size[1]*0.5],
                             [pos[0] + self.size[0]*0.5, pos[1] + self.size[1]*0.5],

                             [pos[0] - self.size[0]*0.5, pos[1] - self.size[1]*0.5],
                             [pos[0] + self.size[0]*0.5, pos[1] + self.size[1]*0.5],
                             [pos[0] - self.size[0]*0.5, pos[1] + self.size[1]*0.5]], dtype=np.float32)

        texCoords = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)

        self.shader = shader2Danitex(vertices, texCoords, texture.img, self.frameCount)

    def handleMouse(self, pos, click):

        if self.inside(pos):
            self.shader.uniforms['ofset'][:] = struct.pack("1f", 1 / self.frameCount)
            if click:
                return self.click
        else:
            self.shader.uniforms['ofset'][:] = struct.pack("1f", 0 / self.frameCount)

        return CONTINUE

    def inside(self, pos):

        for i in (0, 1):
            if pos[i] < (self.pos[i] - self.size[i]*0.5) or pos[i] > (self.pos[i] + self.size[i]*0.5):
                return False
        return True

class dragPlane:

    def __init__(self, ofset, good, audio):

        self.setLocation(ofset)
        vertices = np.array([[-0.01,-0.01,0.004], [0.01,-0.01,0.004], [0.01,0.01,-0.004], [-0.01,-0.01,0.004], [0.01,0.01,-0.004], [-0.01,0.01,-0.004]], dtype=np.float32)
        normals = np.array([[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1]], dtype=np.float32)
        texCoords = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)
        self.value = good
        self.drag = False

        if good:
            texture = material("gfx/3D_model_textures/smooth_audio.png")
        else:
            texture = material("gfx/3D_model_textures/crunchy_audio.png")

        self.shader = shader3D(vertices, normals, texCoords, texture.img)

        self.audio = audio

    def setLocation(self, ofset):

        self.pos = (1-ofset[1]) * (ofset[0] * np.array((610, 480)) + (1-ofset[0]) * np.array((1100, 480))) + ofset[1] * (ofset[0] * np.array((580, 120)) + (1-ofset[0]) * np.array((1120, 120)))
        self.model = np.identity(4)
        self.model[3,0:3] = [-1.465 - ofset[0]*(1.585 - 1.465), 0.4445 + ofset[1]*(0.5223 - 0.4445), 0.990 - ofset[1]*(0.982 - 0.958)]

    def handleMouse(self, pos, drag, relpos, click):

        if self.inside(pos, self.pos, [40, 40]) or (self.drag and drag):
            if click:
                return self.audio
            if drag:
                self.drag = True
                if self.inside(self.pos+relpos, [960,300], [380,200]):
                    self.pos += relpos
                    self.model[3,0:3] += (0.00023, 0.00021, 0.00021) * np.array([relpos[0], -relpos[1], relpos[1]*0.4])
            else:
                self.drag = False

        if self.inside(self.pos, [1230,200], [115, 110]):
            return 1
        elif self.inside(self.pos, [1230,400], [115, 110]):
            return 2
        return 0

    def inside(self, pos, objPos, size):

        for i in (0, 1):
            if pos[i] < (objPos[i] - size[i]) or pos[i] > (objPos[i] + size[i]):
                return False
        return True

    def draw(self, view, camPos):

        self.shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
        self.shader.uniforms['model'][:] = struct.pack('4f4f4f4f', *self.model.flatten())
        self.shader.uniforms['camPos'][:] = struct.pack('3f', *camPos)
        self.shader.render()

class slider:

    def __init__(self, ofset):

        self.click = False
        self.setLocation(ofset)
        vertices = np.array([[0.003363, 0.01, -0.01032], [-0.01042, 0.01, -0.003025], [-0.003367, 0.01, 0.01032], [0.003363, 0.01, -0.01032], [-0.003367, 0.01, 0.01032], [0.01042, 0.01, 0.003031]], dtype=np.float32)
        normals = np.array([[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1]], dtype=np.float32)
        texCoords = np.array([[0,1], [1,1], [1,0], [0,1], [1,0], [0,0]], dtype=np.float32)

        texture = material("gfx/3D_model_textures/slider.png")

        self.shader = shader3D(vertices, normals, texCoords, texture.img)

    def setLocation(self, ofset):

        self.ofset = ofset
        pos = ofset * np.array((1.188,0.596,-1.203)) + (1-ofset) * np.array((1.071,0.596,-1.141))
        self.pos = ofset * np.array((569, 714)) + (1-ofset) * np.array((1350, 694))
        self.model = np.identity(4)
        self.model[3,0:3] = pos

    def handleMouse(self, pos, click, relpos):

        if click and self.inside(pos, self.pos, [50, 60]):
            self.click = True

        if self.click:
            self.ofset = max(0, min(1, self.ofset - 0.0013 * np.array(relpos[0])))
            self.pos = self.ofset * np.array((569, 714)) + (1-self.ofset) * np.array((1350, 694))
            self.model[3,0:3] = self.ofset * np.array((1.172,0.596,-1.195)) + (1-self.ofset) * np.array((1.087,0.596,-1.149))
        return self.ofset

    def inside(self, pos, objPos, size):

        for i in (0, 1):
            if pos[i] < (objPos[i] - size[i]) or pos[i] > (objPos[i] + size[i]):
                return False
        return True

    def draw(self, view, camPos):

        self.shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
        self.shader.uniforms['model'][:] = struct.pack('4f4f4f4f', *self.model.flatten())
        self.shader.uniforms['camPos'][:] = struct.pack('3f', *camPos)
        self.shader.render()

class material:

    def __init__(self, filepath):

        self.pixels = Image.open(filepath).convert("RGBA")
        self.img = ctx.image(self.pixels.size, 'rgba8unorm', self.pixels.tobytes())

class gltfMesh:

    def __init__(self, filename, textures):

        #create the np files with this
        #import precomputeGLTF
        #precomputeGLTF.loadGLTF(filename)

        hasNormals, hasTextures, self.hasJoints, listLenght = np.loadtxt(f"{filename}Data").astype(np.int32)
        self.boundingBox = np.loadtxt(f"{filename}BoundingBox").astype(np.float32)
        self.boundingBox = [[self.boundingBox[2*i], self.boundingBox[2*i + 1]] for i in range(listLenght)]

        indexDataList = [np.loadtxt(f"{filename}IndexDataList{i}").astype(np.int32) for i in range(listLenght)]

        vertexDataList = [np.loadtxt(f"{filename}VertexDataList{i}").astype(np.float32) for i in range(listLenght)]
        vertexDataList = [np.array([vertexDataList[i][3*j:3*j+3] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]

        if hasNormals:
            normalDataList = [np.loadtxt(f"{filename}NormalDataList{i}").astype(np.float32) for i in range(listLenght)]
            normalDataList = [np.array([normalDataList[i][3*j:3*j+3] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]

        if hasTextures:
            texCoordDataList = [np.loadtxt(f"{filename}TexCoordDataList{i}").astype(np.float32) for i in range(listLenght)]
            texCoordDataList = [np.array([texCoordDataList[i][2*j:2*j+2] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]

        if self.hasJoints:
            jointDataList = [np.loadtxt(f"{filename}JointDataList{i}").astype(np.int32) for i in range(listLenght)]
            jointDataList = [np.array([jointDataList[i][4*j:4*j+4] for j in indexDataList[i]], dtype=np.int32) for i in range(listLenght)]
            weightDataList = [np.loadtxt(f"{filename}WeightDataList{i}").astype(np.float32) for i in range(listLenght)]
            weightDataList = [np.array([weightDataList[i][4*j:4*j+4] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]

            self.pose = 0
            nrAnimations, self.timeData = np.loadtxt(f"{filename}MatData").astype(np.int32)
            self.transformMat = [np.loadtxt(f"{filename}Anim{i}Matrices").astype(np.float32) for i in range(nrAnimations)]
            self.nrJoints = len(self.transformMat[0]) // (16 * self.timeData)
            self.transformMat = [[[self.transformMat[anim][i+j*self.nrJoints : i+j*self.nrJoints+16] for i in range(0, 16*self.nrJoints, 16)] for j in range(0, 16*self.timeData, 16)] for anim in range(nrAnimations)]

            self.shaders = [shader3Danimated(vertexDataList[i], normalDataList[i], texCoordDataList[i], jointDataList[i], weightDataList[i], self.nrJoints, textures[i].img) for i in range(listLenght)]
        else:
            self.shaders = [shader3D(vertexDataList[i], normalDataList[i], texCoordDataList[i], textures[i].img) for i in range(listLenght)]

    def setUniform(self):

        for shader in self.shaders:
            animation = np.array(self.transformMat[0][round(self.pose//16)%self.timeData])
            shader.uniforms['animation'][:] = struct.pack(f'{self.nrJoints*16}f', *animation.flatten())

    def draw(self, view, model, camPos):

        for shader in self.shaders:
            shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
            shader.uniforms['model'][:] = struct.pack('4f4f4f4f', *model.flatten())
            shader.uniforms['camPos'][:] = struct.pack('3f', *camPos)
            shader.render()

class billboardMesh:

    def __init__(self, texture):

        self.hasJoints = 0
        vertices = np.array([[0,-0.1,-0.1], [0,-0.1,0.1], [0,0.1,0.1],
                             [0,-0.1,-0.1], [0,0.1,0.1], [0,0.1,-0.1]], dtype=np.float32)
        normals = np.array([[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]], dtype=np.float32)
        texCoords = np.array([[0,0], [1,0], [1,1], [0,0], [1,1], [0,1]], dtype=np.float32)

        self.shaders = [shader3D(vertices, normals, texCoords, texture.img)]

    def draw(self, view, model, camPos):

        for shader in self.shaders:
            shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
            shader.uniforms['model'][:] = struct.pack('4f4f4f4f', *model.flatten())
            shader.uniforms['camPos'][:] = struct.pack('3f', *camPos)
            shader.render()

#####################################################################################

gameObjects = {
    ENTITY_TYPE["player"]: [entity([0,0,0],2),                   gltfMesh("models/vedal987/vedal.gltf",          [material("gfx/3D_model_textures/vedal987.png")])],
    ENTITY_TYPE['keyUI']:  [entity([0,-1,0],2),                  billboardMesh(                                   material("gfx/3D_model_textures//letterE.png"))],
    ENTITY_TYPE["room"]:   [entity([0,0,0],20),                  gltfMesh("models/vedals_room/vedals_room.gltf", [material("gfx/3D_model_textures/floor.png"),
                                                                                                                  material("gfx/3D_model_textures/wall.png"),
                                                                                                                  material("gfx/3D_model_textures/wall.png"),
                                                                                                                  material("gfx/3D_model_textures/wall.png"),
                                                                                                                  material("gfx/3D_model_textures/wall.png"),
                                                                                                                  material("gfx/3D_model_textures/ceiling.png")])],
    ENTITY_TYPE["bed"]:    [entity([-1.4915,0.22373,0.84915],4), gltfMesh("models/bed/bed.gltf",                 [material("gfx/3D_model_textures/bed.png"),
                                                                                                                  material("gfx/3D_model_textures/bed.png")])],
    ENTITY_TYPE["laptop"]: [entity([-1.5,0.4,1.0917],2),         gltfMesh("models/laptop/laptop.gltf",           [material("gfx/3D_model_textures/laptop.png")])],
    ENTITY_TYPE["stool"]:  [entity([1,0.31875,-1.45],2),         gltfMesh("models/stool/stool.gltf",             [material("gfx/3D_model_textures/stool.png")])],
    ENTITY_TYPE["desk"]:   [entity([1.714,0.91399,1.0975],4),    gltfMesh("models/desk/desk.gltf",               [material("gfx/3D_model_textures/desk.png")])],
    ENTITY_TYPE["chair"]:  [entity([0.4528,0.39897,1],2),        gltfMesh("models/chair/chair.gltf",             [material("gfx/3D_model_textures/chair.png")])],
    ENTITY_TYPE["tablet"]: [entity([1.15,0.5925,-1.133],2),      gltfMesh("models/tablet/tablet.gltf",           [material("gfx/3D_model_textures/tablet.png")])]
}

async def main():
    myApp = menu()
    result = CONTINUE
    while result == CONTINUE:
        ctx.new_frame()
        image.clear()
        depth.clear()
        result = myApp.gameLoop()
        if result == NEW_GAME:
            myApp = game()
            result = CONTINUE
        elif result == OPEN_MENU:
            myApp = menu()
            result = CONTINUE
        elif result == CREDITS:
            myApp = game(14)
            result = CONTINUE
        await asyncio.sleep(0)

asyncio.run(main())