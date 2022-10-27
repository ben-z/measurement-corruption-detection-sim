import WebSocketAsPromised from 'websocket-as-promised';
import { mySetInterval, matrixMultiply, generateCircleApproximation, approxeq } from './utils';
import uPlot from 'uplot';
import "uplot/dist/uPlot.min.css";

function ensureSucceeds(res) {
    // Ensures that `res` doesn't have an `error` field
    if (res.error) {
        throw new Error(res.error);
    }

    return res;
}

const egos = {
    ego1: {},
}

// Variable used to expose an API
const bsim_js = global.bsim_js = {};

const WEBSOCKET_OPTIONS = {
    packMessage: data => JSON.stringify(data),
    unpackMessage: data => JSON.parse(data),
    attachRequestId: (data, requestId) => Object.assign({id: requestId}, data), // attach requestId to message as `id` field
    extractRequestId: data => data && data.id,                                  // read requestId from message `id` field
}

async function main() {
    // Initialize world
    const worldSocket = new WebSocketAsPromised('ws://localhost:8765/world', WEBSOCKET_OPTIONS);
    await worldSocket.open();
    console.log("World socket opened: ", worldSocket);
    ensureSucceeds(await worldSocket.sendRequest({command: 'reset'}));
    for (const ego in egos) {
        // ensureSucceeds(await worldSocket.sendRequest({command: `create_entity: ego ${ego} controller=manual`}));
        // ensureSucceeds(await worldSocket.sendRequest({command: `create_entity: ego ${ego} controller=path_following_kmpc`}));
        // ensureSucceeds(await worldSocket.sendRequest({command: `create_entity: ego ${ego} controller=lookahead_lqr`}));
        const target_speed = 5; // m/s
        // target_path = [[-10,3], [10,5], [13,-8], [7, -15], [0,-15], [-10,-3]];
        // target_path = [[20, 20], [20, -20], [-20, -20], [-20, 20]]; // square
        // target_path = [[15, 20], [20, 15], [20, -15], [15, -20], [-15, -20], [-20, -15], [-20, 15], [-15, 20]]; const initial_state = [18,0,-1.5708,target_speed,0]; // square with cut corners
        // target_path = [[-20, 0], [20, 0], [20, 5]]; const initial_state = [0,0,0,0.001,0]; // straight line
        // target_path = [[-20, -20], [20, 20], [-20,30]]; const initial_state = [0,0,0,target_speed,0]; // diagonal line
        const target_path = generateCircleApproximation([0,0], 20, 32).reverse(); const initial_state = [20,0,-1.5708,target_speed,0]; // circle
        const plant_options = {
            initial_state: initial_state,
        }
        // controller = 'manual';
        // controller_options = {};
        const controller = 'lookahead_lqr';
        const controller_options = {
            Q: [
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 10000., 0.],
                [0., 0., 0., 0., 1.],
            ]
        };
        ensureSucceeds(await worldSocket.sendRequest({command: `create_entity: ego ${ego} controller=${controller},target_path=${encodeURIComponent(JSON.stringify(target_path))},controller_options=${encodeURIComponent(JSON.stringify(controller_options))},plant_options=${encodeURIComponent(JSON.stringify(plant_options))},target_speed=${target_speed}`}));
        egos[ego]._socket = new WebSocketAsPromised(`ws://localhost:8765/entities/${ego}`, WEBSOCKET_OPTIONS);
        ensureSucceeds(await egos[ego]._socket.open());
        // testing additive corruption
        // setTimeout(async () => {
        //     const new_state = {
        //         _sensor_state: {
        //             additive_corruption: [0,0,0,-6,0]
        //         }
        //     }
        //     ensureSucceeds(await egos[ego]._socket.sendRequest({command: `update_state: ${JSON.stringify(new_state)}`}));
        // }, 3000)
    }

    const worldCanvas = document.getElementById('worldCanvas');
    worldCanvas.getContext('2d').translate(worldCanvas.width/2, worldCanvas.height/2)
    const rawDebugContainer = document.getElementById('rawDebugContainer');
    const plotContainer = document.getElementById('plotContainer');
    const plots = {};

    let paused = false;
    let errorCount = 0;
    const MAX_ERROR_COUNT = 10;
    const loop = mySetInterval(async () => {
        if (paused) {
            return;
        }
            
        try {
            const worldState = ensureSucceeds(await worldSocket.sendRequest({command: 'tick'})).response;
        
            drawWorld(worldCanvas, worldState);
            drawDebugDashboard(rawDebugContainer, worldState);
            drawPlots(plots, plotContainer, worldState);

        } catch (e) {
            console.error(e);
            displayError(rawDebugContainer, e);
            ++errorCount;
            if (errorCount > MAX_ERROR_COUNT) {
                console.error(`Too many errors, aborting`);
                loop.cancel();
            }
        }
    }, 10);

    bsim_js.tick = () => worldSocket.sendRequest({command: 'tick'}).then(console.log);
    bsim_js.pause = () => {paused = true};
    bsim_js.resume = () => {paused = false};
    bsim_js.isPaused = () => paused;
    bsim_js.getState = () => worldSocket.sendRequest({command: 'state'}).then(console.log);
    bsim_js.resetWorld = () => worldSocket.sendRequest({command: 'reset'}).then(console.log);
    bsim_js.corruptSensorAdditive = (ego, corruption) => egos[ego]._socket.sendRequest({command: `update_state: ${JSON.stringify({_sensor_state: {additive_corruption: corruption}})}`}).then(console.log);
    bsim_js.plots = plots;
}

function drawWorld(canvas, worldState) {
    const ctx = canvas.getContext('2d');

    ctx.clearRect(-canvas.width/2, -canvas.height/2, canvas.width, canvas.height);

    for (const [entityName, entity] of Object.entries(worldState.entities)) {
        switch (entity.type) {
            case 'ego':
                drawVehicle(ctx, entity);
                break;
            default:
                console.error(`Unknown entity type "${entity.type}" for entity "${entityName}"`);
        }
    }
}

function drawDebugDashboard(container, worldState) {
    container.innerHTML = `<pre>${JSON.stringify(worldState, null, 2)}</pre>`;
}

const CURSOR_SYNC_KEY_TIME = 'cursor_sync_key_time';

const PLOT_HORIZON = 10; // seconds

const TIME_SCALE = {
    time: false,
    range: (self, min, max) => bsim_js.isPaused() ? [min, max] : [max - PLOT_HORIZON, max],
}

const MPS_SCALE = {
    range: (self, min, max) => [Math.min(0, min), Math.max(10, max)],
}

const MPS_AXIS = {
    scale: 'm/s',
}

const RAD_SCALE = {
    range: (self, min, max) => [Math.min(-Math.PI, min), Math.max(Math.PI, max)]
}

const DEFAULT_RAD_SPLITS = [-Math.PI, -Math.PI/2, 0, Math.PI/2, Math.PI];

const RAD_AXIS = {
    scale: 'rad',
    values: (self, splits) => {
        if (splits === DEFAULT_RAD_SPLITS) {
            return ['-π', '-π/2', '0', 'π/2', 'π']
        } else {
            return splits.map(s => s.toFixed(2));
        }
    },
    splits: (self, axisIdx, scaleMin, scaleMax) => {
        if (approxeq(scaleMin, -Math.PI) && approxeq(scaleMax, Math.PI)) {
            return DEFAULT_RAD_SPLITS;
        } else {
            return [scaleMin, (scaleMin+scaleMax)/2, scaleMax];
        }
    },
}

const COMMON_PLOT_SETTINGS = {
    cursor: {
        sync: {
            key: CURSOR_SYNC_KEY_TIME,
        },
        drag: {
            x: true,
            y: true,
            uni: 20,
        }
    },
    width: 800,
    height: 200,
}

function drawPlots(plots, container, worldState) {
    const t = worldState.t;

    const PLOT_HORIZON_STEPS = PLOT_HORIZON / worldState.DT;

    for (const [entityName, entity] of Object.entries(worldState.entities)) {
        if (entityName === 'ego1') {
            { // Plots of basic ego state
                const plotID = `${entityName}_basics`;
                if (!plots[plotID]) {
                    const plotContainer = document.createElement('div', {id: plotID});
                    container.appendChild(plotContainer);
                    plots[plotID] = {
                        plot: new uPlot({
                            ...COMMON_PLOT_SETTINGS,
                            title: `${entityName}`,
                            scales: {
                                x: TIME_SCALE,
                                "m/s": MPS_SCALE,
                                "rad": RAD_SCALE,
                            },
                            series: [
                                {
                                    label: 'time (s)',
                                    value: (self, rawValue) => rawValue.toFixed(2),
                                }, 
                                {
                                    label: 'velocity (m/s)',
                                    stroke: "red",
                                    scale: "m/s",
                                    value: (self, rawValue) => rawValue.toFixed(2),
                                },
                                {
                                    label: 'heading (rad)',
                                    stroke: "blue",
                                    scale: "rad",
                                    value: (self, rawValue) => rawValue.toFixed(2),
                                },
                                {
                                    label: 'steering (rad)',
                                    stroke: "green",
                                    scale: "rad",
                                    value: (self, rawValue) => rawValue.toFixed(2),
                                },
                            ],
                            axes: [
                                {},
                                MPS_AXIS,
                                {...RAD_AXIS, side: 1},
                            ]
                        }, [[], [], [], []], plotContainer),
                        data: [[], [], [], []],
                    };
                }
                const plotObj = plots[plotID];
                const vehicleState = decodeVehicleState(entity.state);
                plotObj.data[0] = [...plotObj.data[0].slice(-(PLOT_HORIZON_STEPS -1)), t];
                plotObj.data[1] = [...plotObj.data[1].slice(-(PLOT_HORIZON_STEPS -1)), vehicleState.v];
                plotObj.data[2] = [...plotObj.data[2].slice(-(PLOT_HORIZON_STEPS -1)), vehicleState.theta];
                plotObj.data[3] = [...plotObj.data[3].slice(-(PLOT_HORIZON_STEPS -1)), vehicleState.delta];
                plotObj.plot.setData(plotObj.data);
            }
        }
    }
}

function displayError(container, e) {
    container.innerHTML = `<pre class="alert">Error: ${e.message}\nPlease see the debug console for more info.</pre>`;
}

function drawVehicle(ctx, vehicle) {
    const vehicle_length = vehicle.L; // m
    const tireWidth = 0.2; // m
    const tireLength = 0.5; // m

    const vehicleState = decodeVehicleState(vehicle.state);

    const rearAxleCenter = [vehicleState.x, vehicleState.y];
    const frontAxleCenter = addvector(rearAxleCenter, [vehicle_length * Math.cos(vehicleState.theta), vehicle_length * Math.sin(vehicleState.theta)]);

    ctx.beginPath();
    // draw wheelbase
    ctx.moveTo(...rearAxleCenter.map(m_to_px));
    ctx.lineTo(...frontAxleCenter.map(m_to_px));

    // draw wheels
    ctx.save()
    ctx.translate(m_to_px(rearAxleCenter[0]), m_to_px(rearAxleCenter[1]));
    ctx.rotate(vehicleState.theta)
    ctx.rect(m_to_px(-tireLength/2), m_to_px(-tireWidth/2), m_to_px(tireLength), m_to_px(tireWidth));
    ctx.restore()

    ctx.save()
    ctx.translate(m_to_px(frontAxleCenter[0]), m_to_px(frontAxleCenter[1]));
    ctx.rotate(vehicleState.theta + vehicleState.delta)
    ctx.rect(m_to_px(-tireLength/2), m_to_px(-tireWidth/2), m_to_px(tireLength), m_to_px(tireWidth));
    ctx.restore()

    ctx.stroke();

    // draw target path
    ctx.save()
    ctx.beginPath()
    ctx.setLineDash([m_to_px(0.25), m_to_px(0.5)])
    ctx.moveTo(m_to_px(vehicle.target_path[0][0]), m_to_px(vehicle.target_path[0][1]));
    // rotate the array to the right and draw lines. This gives us a closed path.
    for (const [x, y] of [...vehicle.target_path.slice(1), vehicle.target_path[0]]) {
        ctx.lineTo(m_to_px(x), m_to_px(y));
    }
    ctx.stroke();
    ctx.restore()

    // draw closest path segment
    if (vehicle.controller_debug_output.current_path_segment) {
        const current_path_segment = vehicle.controller_debug_output.current_path_segment;
        ctx.save()
        ctx.beginPath();
        ctx.strokeStyle = 'green';
        ctx.lineWidth = m_to_px(0.2);
        ctx.moveTo(m_to_px(current_path_segment["p0"][0]), m_to_px(current_path_segment["p0"][1]));
        ctx.lineTo(m_to_px(current_path_segment["p1"][0]), m_to_px(current_path_segment["p1"][1]));
        ctx.stroke();
        ctx.restore()
    }

    // draw predicted vehicle trajectory
    if (vehicle.controller_debug_output.predicted_x) {
        const predicted_vehicle_states = vehicle.controller_debug_output.predicted_x.map(decodeVehicleState);

        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = 'blue';
        ctx.fillStyle = 'blue';
        for (const s of predicted_vehicle_states) {
            ctx.arc(m_to_px(s.x), m_to_px(s.y), m_to_px(0.1), 0, 2 * Math.PI);
        }
        ctx.stroke();
        ctx.restore();
    }

    // draw target vehicle trajectory
    if (vehicle.controller_debug_output.target_x) {
        const target_vehicle_states = vehicle.controller_debug_output.target_x.map(decodeVehicleState);

        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = 'darkgreen';
        ctx.fillStyle = 'darkgreen';
        for (const s of target_vehicle_states) {
            ctx.arc(m_to_px(s.x), m_to_px(s.y), m_to_px(0.3), 0, 2 * Math.PI);
        }
        ctx.stroke();
        ctx.restore();
    }
}

main();

function decodeVehicleState(state) {
    return {
        x: state[0],
        y: state[1],
        theta: state[2],
        v: state[3],
        delta: state[4],
    }
}

// https://stackoverflow.com/a/42906936
function addvector(a, b) {
    return a.map((e, i) => e + b[i]);
}

function m_to_px(m) {
    // Meters to pixels
    return m * 10;
}
