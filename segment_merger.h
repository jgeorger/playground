#ifndef SEGMENT_MERGER_H
#define SEGMENT_MERGER_H

struct Segment {
    int start;
    int end;
};

#ifdef __cplusplus
extern "C" {
#endif

void merge_segments(Segment* segments, int* segment_count, int threshold);

#ifdef __cplusplus
}
#endif

#endif // SEGMENT_MERGER_H