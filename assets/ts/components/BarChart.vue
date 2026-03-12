<template>
  <div v-show="barDataIsEmpty" class="alert alert-warning">
    {{ $t("message.noDataToShowInHistogram") }}
  </div>
  <div v-show="!barDataIsEmpty">
    <div class="form-check my-2">
      <input
          type="checkbox"
          class="form-check-input"
          id="checkbox"
          v-model="dateRangeFilteringEnabled"
      />
      <label for="checkbox" class="form-check-label small">
        {{ $t("message.enableDateRangeFiltering") }}
      </label>
    </div>
    <range-slider
        v-if="dataLoaded && dateRangeFilteringEnabled"
        :numberOfMonths="numberOfMonths"
        :initialValues="[selectedRangeStartIndex, selectedRangeEndIndex]"
        :leftMargin="svgStyle.margin.left"
        :rightMargin="svgStyle.margin.right"
        :sliderDisabled="!dateRangeFilteringEnabled"
        @update-value="rangeUpdated"
    >
    </range-slider>

    <svg
        class="d-block mx-auto"
        :width="svgStyle.width"
        :height="svgStyle.height"
    >
      <g
          :transform="`translate(${svgStyle.margin.left}, ${svgStyle.margin.top})`"
      >
        <rect
            class="gbif-alert-bar"
            v-for="(barDataEntry, index) in barData"
            :class="{
          selected:
            !dateRangeFilteringEnabled  ||
            (index >= selectedRangeStartIndex &&
              index <= selectedRangeEndIndex),
        }"
            :key="barDataEntry.yearMonth"
            :x="xScale(barDataEntry.yearMonth)"
            :y="yScale(barDataEntry.count)"
            :width="xScale.bandwidth()"
            :height="svgInnerHeight - yScale(barDataEntry.count)"
        ></rect>

        <g v-yaxis="{ scale: yScale }"/>

        <g :transform="`translate(0, ${svgInnerHeight})`">
          <g v-xaxis="{ scale: xScale, ticks: numberOfXTicks }"/>
        </g>
      </g>
    </svg>
  </div>
</template>

<script setup lang="ts">
import {computed, reactive, ref, watch} from "vue";
import type {DirectiveBinding, ObjectDirective} from "vue";

import {scaleBand, scaleLinear} from "d3-scale";
import {max} from "d3-array";
import {PreparedHistogramDataEntry} from "../interfaces";
import {axisBottom, axisLeft, format, ScaleBand, select} from "d3";
import {DateTime, Interval} from "luxon";
import RangeSlider from "./RangeSlider.vue";

const props = withDefaults(
    defineProps<{
      // Data must be sorted before being passed to the chart
      barData: PreparedHistogramDataEntry[];
      numberOfXTicks?: number;
      dataLoaded?: boolean;
    }>(),
    {numberOfXTicks: 15}
);

const emit = defineEmits<{
  selectedRangeUpdated: [payload: { start: DateTime | null; end: DateTime | null }];
}>();

// --- State ---

const svgStyle = reactive({
  margin: {top: 10, right: 30, bottom: 30, left: 40},
  width: 1116,
  height: 170,
});

const dateRangeFilteringEnabled = ref(false);

// --- Helpers ---

function datetimeToMonthStr(d: DateTime): string {
  return d.year + "-" + d.month;
}

function monthStrToDateTime(m: string): DateTime {
  const split = m.split("-");
  return DateTime.fromObject({
    year: parseInt(split[0]),
    month: parseInt(split[1]),
    day: 1,
  });
}

// Initialized after datetimeToMonthStr is defined
const selectedRangeStart = ref(datetimeToMonthStr(DateTime.now().minus({years: 1})));
const selectedRangeEnd = ref(datetimeToMonthStr(DateTime.now()));

// --- Methods ---

function emitSelectedRange() {
  let rangeStartDate: DateTime | null = null;
  let rangeEndDate: DateTime | null = null;
  if (dateRangeFilteringEnabled.value) {
    rangeStartDate = monthStrToDateTime(selectedRangeStart.value);
    rangeEndDate = monthStrToDateTime(selectedRangeEnd.value);
  }
  emit("selectedRangeUpdated", {start: rangeStartDate, end: rangeEndDate});
}

function rangeUpdated(indexes: number[]) {
  selectedRangeStart.value = props.barData[indexes[0]].yearMonth;
  selectedRangeEnd.value = props.barData[indexes[1]].yearMonth;
  emitSelectedRange();
}

// --- Watchers ---

watch(dateRangeFilteringEnabled, () => {
  emitSelectedRange();
});

// --- Computed ---

const numberOfMonths = computed(() => props.barData.length);

const svgInnerWidth = computed(
    () => svgStyle.width - svgStyle.margin.left - svgStyle.margin.right
);

const svgInnerHeight = computed(
    () => svgStyle.height - svgStyle.margin.top - svgStyle.margin.bottom
);

const endDate = computed(() => DateTime.now());
const startDate = computed(() => endDate.value.minus({month: numberOfMonths.value}));

const xScaleDomain = computed((): string[] => {
  function* months(interval: Interval) {
    let cursor = interval.start!.startOf("month");
    while (cursor < interval.end!) {
      yield cursor;
      cursor = cursor.plus({months: 1});
    }
  }

  const interval = Interval.fromDateTimes(startDate.value, endDate.value);
  return Array.from(months(interval)).map((m: DateTime) => datetimeToMonthStr(m));
});

const xScale = computed((): ScaleBand<string> =>
    scaleBand()
        .range([0, svgInnerWidth.value])
        .paddingInner(0.3)
        .domain(xScaleDomain.value)
);

const dataMax = computed((): number => {
  const maxVal = max(props.barData, (d: PreparedHistogramDataEntry) => d.count);
  return maxVal ?? 0;
});

const yScale = computed(() =>
    scaleLinear().rangeRound([svgInnerHeight.value, 0]).domain([0, dataMax.value])
);

const barDataIsEmpty = computed(() =>
    props.barData.every((e: PreparedHistogramDataEntry) => e.count === 0)
);

const selectedRangeStartIndex = computed(() =>
    props.barData.findIndex((e: PreparedHistogramDataEntry) => e.yearMonth === selectedRangeStart.value)
);

const selectedRangeEndIndex = computed(() =>
    props.barData.findIndex((e: PreparedHistogramDataEntry) => e.yearMonth === selectedRangeEnd.value)
);

// --- Directives ---
// In <script setup>, variables prefixed with `v` are automatically available as directives in the template.
// vYaxis → v-yaxis, vXaxis → v-xaxis

const vYaxis: ObjectDirective<SVGGElement> = {
  beforeUpdate(el, binding: DirectiveBinding): void {
    const scaleFunction = binding.value.scale;
    const yAxisTicks = scaleFunction
        .ticks(4)
        .filter((tick: number) => Number.isInteger(tick));
    const yAxis = axisLeft<number>(scaleFunction)
        .tickValues(yAxisTicks)
        .tickFormat(format("d"));
    yAxis(select(el));
  },
};

const vXaxis: ObjectDirective<SVGGElement> = {
  beforeUpdate(el, binding: DirectiveBinding): void {
    const scaleFunction = binding.value.scale;
    const numberOfTicks = binding.value.ticks;
    const numberOfElems = scaleFunction.domain().length;
    const moduloVal = Math.floor(numberOfElems / numberOfTicks);
    const d3Axis = axisBottom<string>(scaleFunction).tickValues(
        scaleFunction.domain().filter((_: string, i: number) => !(i % moduloVal))
    );
    d3Axis(select(el));
  },
};
</script>

<style scoped>
.gbif-alert-bar {
  fill: #00a58d;
  opacity: 0.3;
}

.selected {
  fill: #198754 !important;
  opacity: 1;
}
</style>
