	W?/?'V@W?/?'V@!W?/?'V@	?R??Ǖ???R??Ǖ??!?R??Ǖ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$W?/?'V@??? ?r??A4??@?ZV@Y??	h"l??*	     }@2F
Iterator::Model??\m????!????V@)????H??1Dr؃V@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ?????!?љT;:@)/n????1|????6@:Preprocessing2U
Iterator::Model::ParallelMapV2A??ǘ???!Dr؃H@)A??ǘ???1Dr؃H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???߾??!?'|???@)?5?;Nс?1O?n?	???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?e??a???!y?wO?"@) ?o_?y?1-?????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!?V???*??){?G?zt?1?V???*??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?s?!r؃H{??)a2U0*?s?1r؃H{??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX9??v???!#?e?
@)??H?}]?1???l????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?R??Ǖ??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??? ?r????? ?r??!??? ?r??      ??!       "      ??!       *      ??!       2	4??@?ZV@4??@?ZV@!4??@?ZV@:      ??!       B      ??!       J	??	h"l????	h"l??!??	h"l??R      ??!       Z	??	h"l????	h"l??!??	h"l??JCPU_ONLYY?R??Ǖ??b 