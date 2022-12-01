import WebSocketAsPromised from 'websocket-as-promised';
import { mySetInterval, matrixMultiply, generateCircleApproximation, approxeq, predSlice, frenet2global_path } from './utils';
import uPlot from 'uplot';
import "uplot/dist/uPlot.min.css";

const ENTITY_COLOR_MAP = [
    {
        primary: 'green',
        secondary: 'blue',
        error: 'red',
        palette: [
            // from dark to light
            // https://www.vecteezy.com/vector-art/2681487-green-color-palette-with-hex
            '#3A6324',
            '#267D39',
            '#329542',
            '#63A355',
            '#B7D2B6',
            '#CDE0CD',
            '#E1EDE0',
        ]
    },
    {
        primary: 'blue',
        secondary: 'green',
        error: 'red',
        palette: [
            // from dark to light
            // https://www.vecteezy.com/vector-art/2681487-green-color-palette-with-hex
            '#3A6324',
            '#267D39',
            '#329542',
            '#63A355',
            '#B7D2B6',
            '#CDE0CD',
            '#E1EDE0',
        ]
    },
]

function ensureSucceeds(res) {
    // Ensures that `res` doesn't have an `error` field
    if (res.error) {
        throw new Error(res.error);
    }

    return res;
}

const egos = {
    ego1: {},
    ego2: {},
}

function initializeEgoConfig() {
    const BASIC_TARGET_SPEED = 5; // m/s
    const REF_SQUARE = {
        global_ref_path: [[20, 20], [20, -20], [-20, -20], [-20, 20]],
        get_initial_state: (target_speed) => [0, 0, 0, target_speed, 0],
    };
    const REF_SQUARE_WITH_CUT_CONERS = {
        global_ref_path: [[15, 20], [20, 15], [20, -15], [15, -20], [-15, -20], [-20, -15], [-20, 15], [-15, 20]],
        get_initial_state: (target_speed) => [18,0,-1.5708, target_speed,0],
    };
    const REF_STRAIGHT_LINE = {
        global_ref_path: [[-20, 0], [20, 0], [20, 5]],
        get_initial_state: (target_speed) => [0,0,0, target_speed,0],
    };
    const REF_DIAGONAL_LINE = {
        global_ref_path: [[-20, -20], [20, 20], [-20,30]],
        get_initial_state: (target_speed) => [0,0,0, target_speed,0],
    };
    const REF_CIRCLE_32_DIV = {
        global_ref_path: generateCircleApproximation([0,0], 20, 32).reverse(),
        get_initial_state: (target_speed) => [20,0,-1.5708, target_speed,-0.145],
    };
    const REF_FRENET_GENERATD_STRAIGHT_LINE = (() => {
        const slist = [...Array(100).keys()].map(i => i);
        const dlist = new Array(slist.length).fill(0);

        return {
            global_ref_path: frenet2global_path([[-20, 0], [20, 0], [20, 5]], slist, dlist),
            get_initial_state: (target_speed) => [0,0,0, target_speed,0],
        };
    })();
    const REF_FRENET_GENERATD_CIRCLE = (() => {
        const slist = [...Array(100).keys()].map(i => i);   
        const dlist = new Array(slist.length).fill(0);

        return {
            global_ref_path: frenet2global_path(generateCircleApproximation([0,0], 20, 32).reverse(), slist, dlist),
            get_initial_state: (target_speed) => [20,0,-1.5708, target_speed,-0.145],
        };
    })();
    const REF_NOT_SURE = {
        global_ref_path: [[-10,3], [10,5], [13,-8], [7, -15], [0,-15], [-10,-3]],
        get_initial_state: (target_speed) => [0,0,0, target_speed,0],
    }

    const BASIC_CONTROLLER_CONFIG = {
        controller: 'lookahead_lqr',
        controller_options: {
            Q: [
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 10000., 0.],
                [0., 0., 0., 0., 1.],
            ]
        }
    }
    const BASIC_PLANNER_CONFIG = {
        planner: 'lateral_profile',
        planner_options: {
            lateral_deviation_profile: {
                'interpolation': 'linear',
                'periodic': true,
                'points': [
                    { 't': 0, 'lateral_deviation': 0 },
                    { 't': 5, 'lateral_deviation': 0 },
                    { 't': 10, 'lateral_deviation': 3 },
                    { 't': 20, 'lateral_deviation': 3 },
                    { 't': 30, 'lateral_deviation': 0 },
                    { 't': 35, 'lateral_deviation': 0 },
                    { 't': 40, 'lateral_deviation': -3 },
                    { 't': 50, 'lateral_deviation': -3 },
                    { 't': 60, 'lateral_deviation': 0 },
                ],
            }
        }
    };

    { // ego1
        const ego = egos.ego1;
        const ref = REF_CIRCLE_32_DIV;
        const target_speed = BASIC_TARGET_SPEED;

        ego.controller = BASIC_CONTROLLER_CONFIG.controller;
        ego.controller_options = BASIC_CONTROLLER_CONFIG.controller_options;
        ego.global_ref_path = ref.global_ref_path;
        ego.plant_options = {
            initial_state: ref.get_initial_state(target_speed),
        };
        ego.target_speed = target_speed;
        ego.planner = BASIC_PLANNER_CONFIG.planner;
        ego.planner_options = BASIC_PLANNER_CONFIG.planner_options;
        ego.detector = 'l1_optimizer';
    }

    { // ego2
        const ego = egos.ego2;
        const ref = REF_CIRCLE_32_DIV;
        const target_speed = BASIC_TARGET_SPEED; // m/s

        ego.controller = BASIC_CONTROLLER_CONFIG.controller;
        ego.controller_options = BASIC_CONTROLLER_CONFIG.controller_options;
        ego.global_ref_path = ref.global_ref_path;
        ego.plant_options = {
            initial_state: ref.get_initial_state(target_speed),
        };
        ego.target_speed = target_speed;
        ego.planner = BASIC_PLANNER_CONFIG.planner;
        ego.planner_options = BASIC_PLANNER_CONFIG.planner_options;
        ego.detector = 'none';
    }
}

// Variable used to expose an API
const bsim_js = global.bsim_js = {};

const WEBSOCKET_OPTIONS = {
    packMessage: data => JSON.stringify(data),
    unpackMessage: data => JSON.parse(data),
    attachRequestId: (data, requestId) => Object.assign({id: requestId}, data), // attach requestId to message as `id` field
    extractRequestId: data => data && data.id,                                  // read requestId from message `id` field
}

function entityToPlotContainerID(entityName) {
    // Converts an entity name to a plot container ID
    switch (entityName) {
        case 'ego1':
            return 'plotContainer1';
        case 'ego2':
            return 'plotContainer2';
        default:
            return 'plotContainerGlobal';
    }
}

async function main() {
    // Initialize world
    const worldSocket = new WebSocketAsPromised('ws://localhost:8765/world', WEBSOCKET_OPTIONS);
    await worldSocket.open();
    console.log("World socket opened: ", worldSocket);
    ensureSucceeds(await worldSocket.sendRequest({command: 'reset'}));
    initializeEgoConfig();
    for (const [egoName, egoConfig] of Object.entries(egos)) {
        ensureSucceeds(await worldSocket.sendRequest({
            command: `create_entity: ego ${egoName} ${encodeURIComponent(JSON.stringify(egoConfig))}`
        }));
        egos[egoName]._socket = new WebSocketAsPromised(`ws://localhost:8765/entities/${egoName}`, WEBSOCKET_OPTIONS);
        ensureSucceeds(await egos[egoName]._socket.open());
    }

    const bevCanvas = document.getElementById('bevCanvas');
    bevCanvas.getContext('2d').translate(bevCanvas.width/2, bevCanvas.height/2)
    const rawDebugContainer = document.getElementById('rawDebugContainer');
    const plotContainers = {
        plotContainerGlobal: document.getElementById('plotContainerGlobal'),
        plotContainer1: document.getElementById('plotContainer1'),
        plotContainer2: document.getElementById('plotContainer2'),
    }

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
        
            if (Object.keys(worldState.entities).length > ENTITY_COLOR_MAP.length) {
                throw new Error("Please update ENTITY_COLOR_MAP to accomodate the additional entities.");
            }
        
            bsim_js.pushWorldState(worldState);
            drawBEV(bevCanvas, worldState);
            drawDebugDashboard(rawDebugContainer, worldState);
            drawPlots(plots, plotContainers, worldState);
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
    bsim_js.corruptSensorMultiplicative = (ego, corruption) => egos[ego]._socket.sendRequest({ command: `update_state: ${JSON.stringify({ _sensor_state: { multiplicative_corruption: corruption}})}`}).then(console.log);
    bsim_js.plots = plots;
    
    // bsim settings
    let plot_horizon = 10; // seconds
    bsim_js.get_plot_horizon = () => plot_horizon;
    bsim_js.set_plot_horizon = new_horizon => { plot_horizon = new_horizon; };
    bsim_js.get_data_horizon = () => 10 * bsim_js.get_plot_horizon(); // seconds

    let worldStates = [];
    bsim_js.getBEVCanvas = () => bevCanvas;
    bsim_js.getRawDebugContainer = () => rawDebugContainer;
    bsim_js.getWorldStates = () => worldStates;
    bsim_js.pushWorldState = s => {
        worldStates = [
            ...sliceToHorizon([worldStates], worldStates.map(st => st.t), s.t, bsim_js.get_data_horizon())[0],
            s
        ];
    }
}

function getOrCreateBEVToggleState(toggle_id, toggle_label, toggle_default) {
    toggle_id = `bev-toggle-${toggle_id}`;
    let toggle = document.getElementById(toggle_id);
    if (!toggle) {
        const container = document.createElement('div');
        toggle = document.createElement('input');
        toggle.type = 'checkbox';
        toggle.id = toggle_id;
        toggle.checked = toggle_default;
        const label = document.createElement('label');
        label.htmlFor = toggle_id;
        label.appendChild(document.createTextNode(toggle_label));
        container.appendChild(toggle);
        container.appendChild(label);
        document.getElementById('bevToggleContainer').appendChild(container);
    }
    return toggle.checked;
}

function drawBEV(canvas, worldState) {
    /*
    Draws the BEV of the world
    */
    const ctx = canvas.getContext('2d');

    ctx.clearRect(-canvas.width/2, -canvas.height/2, canvas.width, canvas.height);

    ctx.font = '800 13px Courier New';
    ctx.fillText(`t=${worldState.t.toFixed(2)}s`, -canvas.width/2 + 10, -canvas.height/2 + 20);

    for (const [entityName, entity] of Object.entries(worldState.entities)) {
        switch (entity.type) {
            case 'ego':
                drawVehicle(ctx, entityName, entity);
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

const TIME_SCALE = {
    time: false,
    range: (self, min, max) => self.bsim_hasSelect ? [min, max] : [max - bsim_js.get_plot_horizon(), max],
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

const LATDEV_SCALE = {
    range: (self, min, max) => [Math.min(-5, min), Math.max(5, max)],
}

const LATDEV_AXIS = {
    scale: 'latdev',
}

const EXECUTION_TIME_SCALE = {
    range: (self, min, max) => [0, 0.05],
}

const EXECUTION_TIME_AXIS = {
    scale: 'execution_time',
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
        },
        // lock: true, // FIXME: this breaks double-click auto-scale
        dataIdx: (self, seriesIdx, hoveredIdx, cursorXVal) => {
            // find the closest non-null data point. Taken from
            // https://github.com/leeoniya/uPlot/blob/5bebae5/demos/nearest-non-null.html#L55-L88
            let xValues = self.data[0];
            let yValues = self.data[seriesIdx];

            if (yValues[hoveredIdx] == null) {
                let nonNullLft = null,
                    nonNullRgt = null,
                    i;

                i = hoveredIdx;
                while (nonNullLft == null && i-- > 0) {
                    if (yValues[i] != null)
                        nonNullLft = i;
                }

                i = hoveredIdx;
                while (nonNullRgt == null && i++ < yValues.length) {
                    if (yValues[i] != null)
                        nonNullRgt = i;
                }

                let rgtVal = nonNullRgt == null ? Infinity : xValues[nonNullRgt];
                let lftVal = nonNullLft == null ? -Infinity : xValues[nonNullLft];

                let lftDelta = cursorXVal - lftVal;
                let rgtDelta = rgtVal - cursorXVal;

                hoveredIdx = lftDelta <= rgtDelta ? nonNullLft : nonNullRgt;
            }

            return hoveredIdx;
        }
    },
    width: 800,
    height: 200,
    hooks: {
        'setSelect': [
            self => {
                self.bsim_hasSelect = self.select.width > 0 || self.select.height > 0;
            }
        ],
        'setScale': [
            self => {
                self.bsim_hasSelect = false;
            }
        ],
        'setCursor': [
            self => {
                // Update BEV based on cursor position
                if (!bsim_js.isPaused() || !self.cursor.idx) {
                    return;
                }

                drawBEV(bsim_js.getBEVCanvas(), bsim_js.getWorldStates()[self.cursor.idx]);
                drawDebugDashboard(bsim_js.getRawDebugContainer(), bsim_js.getWorldStates()[self.cursor.idx]);
            }
        ],
    }
}

function sliceToHorizon(arrs, tarr, t, horizon) {
    for (const arr of arrs) {
        if (arr.length !== tarr.length) {
            throw new Error(`sliceToHorizon: arr.length (${arr.length}) !== tarr.length (${tarr.length})`);
        }
    }
    const res_tarr = predSlice(tarr, e => e >= t - horizon);
    // TODO: this takes a long time. Replace this with a ring buffer.
    // Then we don't need this function at all.
    // When we need to resize (i.e. when we change the horizon),
    // we simply need to recreate the ring buffer.
    const res_arrs = arrs.map(arr => arr.slice(-res_tarr.length));
    return res_arrs
}

const EXECUTION_TIME_FIELDS = [
    { 'name': 'total', 'field': 'request_handler' },
    // { 'name': 'sensor', 'field': 'sensor' },
    { 'name': 'detector', 'field': 'detector' },
    // { 'name': 'estimator', 'field': 'estimator' },
    { 'name': 'planner', 'field': 'planner' },
    { 'name': 'controller', 'field': 'controller' },
    // { 'name': 'plant', 'field': 'plant' },
]

function drawPlots(plots, containers, worldState) {
    const t = worldState.t;

    {
        const container = containers[entityToPlotContainerID('global')]
        const plotID = `execution_time`;
        if (!plots[plotID]) {
            const onePlotContainer = document.createElement('div', {id: plotID});
            container.appendChild(onePlotContainer);
            const plot_settings = {
                ...COMMON_PLOT_SETTINGS,
                title: `Execution Time (s)`,
                scales: {
                    x: TIME_SCALE,
                    execution_time: EXECUTION_TIME_SCALE,
                },
                series: [
                    {
                        label: 'time (s)',
                        value: (self, rawValue) => rawValue.toFixed(2),
                    }, 
                    {
                        label: 'World (s)',
                        stroke: "blue",
                        scale: "execution_time",
                        value: (self, rawValue) => rawValue == null ? "-" : rawValue.toFixed(3),
                    },
                    ...Object.entries(worldState.entities).map(([entityName, entity], i) =>
                        EXECUTION_TIME_FIELDS.map(({name, _field}, j) => ({
                            label: `${entityName} ${name} (s)`,
                            stroke: ENTITY_COLOR_MAP[i].palette[j],
                            scale: "execution_time",
                            value: (self, rawValue) => rawValue == null ? "-" : rawValue.toFixed(3),
                        }))
                    ).flat()
                ],
                axes: [
                    {},
                    EXECUTION_TIME_AXIS,
                ]
            }
            plots[plotID] = {
                plot: new uPlot(plot_settings, Array(plot_settings.series.length).fill([]), onePlotContainer),
                data: Array(plot_settings.series.length).fill([]),
            };
        }
        const plotObj = plots[plotID];
        plotObj.data = sliceToHorizon(plotObj.data, plotObj.data[0], t, bsim_js.get_data_horizon());
        plotObj.data[0].push(t);
        plotObj.data[1].push(worldState.execution_times.request_handler);
        const entityStartIdx = 2;
        for (const [[_entityName, entity], i] of Object.entries(worldState.entities).map((e,i)=>[e, i+entityStartIdx])) {
            for (const [{_name, field}, j] of EXECUTION_TIME_FIELDS.map((e,j) => [e, j])) {
                plotObj.data[entityStartIdx + (i-entityStartIdx) * EXECUTION_TIME_FIELDS.length + j].push(entity.execution_times[field]);
            }
        }
        plotObj.plot.setData(plotObj.data);
    }

    for (const [entityName, entity] of Object.entries(worldState.entities)) {
        const container = containers[entityToPlotContainerID(entityName)]
        if (entityName.match('^ego\\d+$')) {
            { // Plots of basic ego state
                const plotID = `${entityName}_basics`;
                if (!plots[plotID]) {
                    const onePlotContainer = document.createElement('div', {id: plotID});
                    container.appendChild(onePlotContainer);
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
                                {...RAD_AXIS, side: 1, grid: {show: false}},
                            ]
                        }, [[], [], [], []], onePlotContainer),
                        data: [[], [], [], []],
                    };
                }
                const plotObj = plots[plotID];
                const vehicleState = decodeVehicleState(entity.state);
                plotObj.data = sliceToHorizon(plotObj.data, plotObj.data[0], t, bsim_js.get_data_horizon());
                plotObj.data[0].push(t);
                plotObj.data[1].push(vehicleState.v);
                plotObj.data[2].push(vehicleState.theta);
                plotObj.data[3].push(vehicleState.delta);
                plotObj.plot.setData(plotObj.data);
            }
            {
                const plotID = `${entityName}_detector`;
                if (!plots[plotID]) {
                    const onePlotContainer = document.createElement('div', {id: plotID});
                    container.appendChild(onePlotContainer);
                    plots[plotID] = {
                        plot: new uPlot({
                            ...COMMON_PLOT_SETTINGS,
                            title: `${entityName} Detector Debug`,
                            scales: {
                                x: TIME_SCALE,
                                idx: {
                                    range: (self, min, max) => [Math.min(0,min), Math.max(32, max)],
                                    
                                },
                                error: {
                                    range: (self, min, max) => [Math.min(-1,min), Math.max(1,max)],
                                },
                            },
                            series: [
                                {
                                    label: 'time (s)',
                                    value: (self, rawValue) => rawValue.toFixed(2),
                                }, 
                                {
                                    label: 'Path Memory Index',
                                    stroke: "blue",
                                    scale: "idx",
                                    paths: uPlot.paths.stepped({align: 1}),
                                    value: (self, rawValue) => rawValue == null ? "-" : rawValue.toFixed(0),
                                },
                                {
                                    label: 'State estimation l2 error (x0)',
                                    stroke: "red",
                                    scale: "error",
                                    value: (self, rawValue) => rawValue == null ? "-" : rawValue.toFixed(2),
                                    points: {
                                        space: 0,
                                    }
                                },
                                {
                                    label: 'State estimation l2 error (xf)',
                                    stroke: "green",
                                    scale: "error",
                                    value: (self, rawValue) => rawValue == null ? "-" : rawValue.toFixed(2),
                                    points: {
                                        space: 0,
                                    }
                                },
                            ],
                            axes: [
                                {},
                                {
                                    scale: 'idx',
                                    grid: {
                                        show: false,
                                    },
                                },
                                {
                                    scale: 'error',
                                    side: 1,
                                }
                            ]
                        }, [[], [], [], []], onePlotContainer),
                        data: [[], [], [], []],
                    };
                }
                const plotObj = plots[plotID];
                plotObj.data = sliceToHorizon(plotObj.data, plotObj.data[0], t, bsim_js.get_data_horizon());
                plotObj.data[0].push(t);
                plotObj.data[1].push(entity.detector_debug_output.path_memory_segment_idx);
                plotObj.data[2].push(entity.detector_debug_output.state_estimation_l2_error_x0);
                plotObj.data[3].push(entity.detector_debug_output.state_estimation_l2_error_xf);
                plotObj.plot.setData(plotObj.data);
            }
            {
                const plotID = `${entityName}_detector_2`;
                if (!plots[plotID]) {
                    const onePlotContainer = document.createElement('div', {id: plotID});
                    container.appendChild(onePlotContainer);
                    plots[plotID] = { onePlotContainer }
                }
                if (!plots[plotID].plot && entity.detector_debug_output.sensor_validity_map) {
                    const { onePlotContainer } = plots[plotID];

                    const plot_settings = {
                        ...COMMON_PLOT_SETTINGS,
                        title: `${entityName} Detector Debug 2`,
                        scales: {
                            x: TIME_SCALE,
                            true_false: {
                                range: (self, min, max) => [Math.min(0,min), Math.max(1, max)],
                            },
                        },
                        series: [
                            {
                                label: 'time (s)',
                                value: (self, rawValue) => rawValue.toFixed(2),
                            },
                            ...entity.detector_debug_output.sensor_validity_map.map(
                                (sensor_validity, sensor_idx) => ({
                                    label: `Sensor ${sensor_idx} Validity`,
                                    scale: 'true_false',
                                    stroke: 'blue',
                                })
                            ).flat()
                        ],
                        axes: [
                            {},
                            {
                                scale: 'true_false',
                            },
                        ]
                    }
                    plots[plotID].plot = new uPlot(plot_settings, Array(plot_settings.series.length).fill([]), onePlotContainer);
                    plots[plotID].data = Array(plot_settings.series.length).fill([]);
                }
                if (plots[plotID].plot) {
                    const plotObj = plots[plotID];
                    plotObj.data = sliceToHorizon(plotObj.data, plotObj.data[0], t, bsim_js.get_data_horizon());

                    plotObj.data = sliceToHorizon(plotObj.data, plotObj.data[0], t, bsim_js.get_data_horizon());
                    plotObj.data[0].push(t);
                    const sensorStartIdx = 1;
                    entity.detector_debug_output.sensor_validity_map.forEach((sensor_validity, sensor_idx) => {
                        plotObj.data[sensorStartIdx + sensor_idx].push(sensor_validity);
                    });
                    plotObj.plot.setData(plotObj.data);
                }
            }
            {
                const plotID = `${entityName}_planner`;
                if (!plots[plotID]) {
                    const onePlotContainer = document.createElement('div', {id: plotID});
                    container.appendChild(onePlotContainer);
                    const plot_settings = {
                        ...COMMON_PLOT_SETTINGS,
                        title: `${entityName} Planner Debug`,
                        scales: {
                            x: TIME_SCALE,
                            latdev: LATDEV_SCALE,
                        },
                        series: [
                            {
                                label: 'time (s)',
                                value: (self, rawValue) => rawValue.toFixed(2),
                            }, 
                            {
                                label: 'Target Lateral Deviation',
                                stroke: "blue",
                                scale: "latdev",
                                value: (self, rawValue) => rawValue == null ? "-" : rawValue.toFixed(2),
                            },
                        ],
                        axes: [
                            {},
                            LATDEV_AXIS,
                        ]
                    }
                    plots[plotID] = {
                        plot: new uPlot(plot_settings, Array(plot_settings.series.length).fill([]), onePlotContainer),
                        data: Array(plot_settings.series.length).fill([]),
                    };
                }
                const plotObj = plots[plotID];
                plotObj.data = sliceToHorizon(plotObj.data, plotObj.data[0], t, bsim_js.get_data_horizon());
                plotObj.data[0].push(t);
                plotObj.data[1].push(entity.planner_debug_output.target_lateral_deviations?.[0]);
                plotObj.plot.setData(plotObj.data);
            }
        }
    }
}

function displayError(container, e) {
    container.innerHTML = `<pre class="alert">Error: ${e.message}\nPlease see the debug console for more info.</pre>`;
}

function drawVehicle(ctx, vehicleName, vehicle) {
    const vehicle_length = vehicle.L; // m
    const tireWidth = 0.2; // m
    const tireLength = 0.5; // m

    const vehicleState = decodeVehicleState(vehicle.state);

    if (getOrCreateBEVToggleState(`${vehicleName}_state`, `${vehicleName} State`, true)) {
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
    }

    // draw global reference path
    const global_ref_path = vehicle.global_ref_path;
    if (global_ref_path && getOrCreateBEVToggleState(`${vehicleName}-global_ref_path`, `${vehicleName} global reference path`, true)) {
        const is_closed = true;
        ctx.save()
        ctx.beginPath()
        ctx.strokeStyle = 'gray';
        ctx.fillStyle = 'gray';
        ctx.setLineDash([m_to_px(0.05), m_to_px(0.5)])
        ctx.moveTo(m_to_px(global_ref_path[0][0]), m_to_px(global_ref_path[0][1]));
        // rotate the array to the right and draw lines. This gives us a closed path.
        for (const [x, y] of global_ref_path.slice(1).concat(is_closed ? [global_ref_path[0]] : [])) {
            ctx.lineTo(m_to_px(x), m_to_px(y));
        }
        ctx.stroke();
        ctx.restore()
        ctx.save()
        for (const [x, y] of global_ref_path) {
            ctx.beginPath()
            ctx.strokeStyle = 'gray';
            ctx.fillStyle = 'gray';
            ctx.arc(m_to_px(x), m_to_px(y), m_to_px(0.2), 0, 2 * Math.PI);
            ctx.stroke();
        }
        ctx.restore()
    }

    // draw path memory
    const path_memory = vehicle.detector_debug_output.path_memory;
    if (path_memory && getOrCreateBEVToggleState(`${vehicleName}-detector_path_memory`, `${vehicleName} detector path memory`, true)) {
        ctx.save()
        ctx.beginPath()
        ctx.strokeStyle = 'gray';
        ctx.fillStyle = 'gray';
        ctx.setLineDash([m_to_px(0.05), m_to_px(0.5)])
        ctx.moveTo(m_to_px(path_memory[0][0]), m_to_px(path_memory[0][1]));
        // rotate the array to the right and draw lines. This gives us a closed path.
        for (const [x, y] of path_memory.slice(1).concat(IS_CLOSED_PATH ? [path_memory[0]] : [])) {
            ctx.lineTo(m_to_px(x), m_to_px(y));
        }
        ctx.stroke();
        ctx.restore()
        ctx.save()
        for (const [x, y] of path_memory) {
            ctx.beginPath()
            ctx.strokeStyle = 'gray';
            ctx.fillStyle = 'gray';
            ctx.arc(m_to_px(x), m_to_px(y), m_to_px(0.2), 0, 2 * Math.PI);
            ctx.stroke();
        }
        ctx.restore()
    }

    // TODO: dynamically enable and disable plots
    // draw target path
    const target_path = vehicle.planner_output.target_path;
    const IS_CLOSED_PATH = false;
    if (target_path && getOrCreateBEVToggleState(`${vehicleName}-target_path`, `${vehicleName} target path`, true)) {
        ctx.save()
        ctx.beginPath()
        ctx.setLineDash([m_to_px(0.25), m_to_px(0.5)])
        ctx.moveTo(m_to_px(target_path[0][0]), m_to_px(target_path[0][1]));
        // rotate the array to the right and draw lines. This gives us a closed path.
        for (const [x, y] of target_path.slice(1).concat(IS_CLOSED_PATH ? [target_path[0]] : [])) {
        // for (const [x, y] of [...target_path.slice(1), target_path[0]]) {
            ctx.lineTo(m_to_px(x), m_to_px(y));
        }
        ctx.stroke();
        ctx.restore()
        ctx.save()
        for (const [x, y] of target_path) {
        // for (const [x, y] of [...target_path.slice(1), target_path[0]]) {
            ctx.beginPath()
            ctx.arc(m_to_px(x), m_to_px(y), m_to_px(0.2), 0, 2 * Math.PI);
            ctx.stroke();
        }
        ctx.restore()
    }

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
    if (vehicle.controller_debug_output.target_x && getOrCreateBEVToggleState(`${vehicleName}-controller_target`, `${vehicleName} controller target`, true)) {
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

    // draw planner lookahead point
    if (vehicle.planner_debug_output.lookahead_point && getOrCreateBEVToggleState(`${vehicleName}-planner_lookahead_point`, `${vehicleName} planner lookahead point`, true)) {
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = 'lightgreen';
        ctx.fillStyle = 'lightgreen';
        ctx.arc(m_to_px(vehicle.planner_debug_output.lookahead_point[0]), m_to_px(vehicle.planner_debug_output.lookahead_point[1]), m_to_px(0.3), 0, 2 * Math.PI);
        ctx.stroke();
        ctx.restore();
    }

    // draw planner lookbehind point
    if (vehicle.planner_debug_output.lookbehind_point && getOrCreateBEVToggleState(`${vehicleName}-planner_lookbehind_point`, `${vehicleName} planner lookbehind point`, false)) {
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = 'lightgreen';
        ctx.fillStyle = 'lightgreen';
        ctx.arc(m_to_px(vehicle.planner_debug_output.lookbehind_point[0]), m_to_px(vehicle.planner_debug_output.lookbehind_point[1]), m_to_px(0.3), 0, 2 * Math.PI);
        ctx.stroke();
        ctx.restore();
    }

    // draw true states of the vehicle used in the detector
    if (vehicle.detector_debug_output.true_states) {
        const true_vehicle_states = vehicle.detector_debug_output.true_states.map(decodeVehicleState);

        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = 'red';
        ctx.fillStyle = 'red';
        for (const s of true_vehicle_states) {
            ctx.arc(m_to_px(s.x), m_to_px(s.y), m_to_px(0.2), 0, 2 * Math.PI);
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
