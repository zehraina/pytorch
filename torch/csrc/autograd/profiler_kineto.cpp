#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/profiler_kineto.h>

#include <c10/macros/Export.h>
#include <c10/util/C++17.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/overloaded.h>
#include <c10/util/variant.h>

#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/nvtx_observer.h>

#include <ATen/Context.h>

#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef USE_KINETO
#include <libkineto.h>
#include <time_since_epoch.h>

#ifndef _MSC_VER
// TODO: TO be removed, once this properly works from libkineto
// Literal copy-n-paste from third_party/kineto/libkineto/src/WeakSymbols.cpp
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library
// when cuPTI is linked statically For more details see
// https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error(
      "Dummy implementation of acc_get_device_type is not supposed to be called!");
}
} // extern "C"
#endif // _MSC_VER
#endif // USE_KINETO

namespace torch {
namespace autograd {
namespace profiler {

namespace {
inline int64_t getTimeUs() {
#ifdef USE_KINETO
  return libkineto::timeSinceEpoch(std::chrono::system_clock::now());
#else
  return torch::profiler::impl::getTime() / 1000;
#endif // USE_KINETO
}

using torch::profiler::impl::ActiveProfilerType;
using torch::profiler::impl::dtypesToStr;
using torch::profiler::impl::EventType;
using torch::profiler::impl::ExtraFields;
using torch::profiler::impl::ProfilerThreadLocalStateBase;
using torch::profiler::impl::Result;
using torch::profiler::impl::shapesToStr;
using torch::profiler::impl::stacksToStr;
using torch::profiler::impl::kineto::annotation_t;

struct EventFieldsVisitor {
  EventFieldsVisitor(
      std::shared_ptr<Result>& result,
      KinetoEvent& kineto_event,
      const post_process_t& post_process)
      : kineto_event_{kineto_event}, post_process_{post_process} {
    pushPythonMetadata(result->parent_.lock());
    c10::visit(*this, result->extra_fields_);
    handleStack(result->parent_);
  }

  void operator()(ExtraFields<EventType::TorchOp>& op_event) {
    handleJIT(op_event);
    kineto_event_.get()
        .endThreadId(op_event.end_tid_)
        .scope((int8_t)op_event.scope_)
        .debugHandle(op_event.debug_handle_)
        .setAsync(op_event.is_async_);

    auto& shapes = op_event.inputs_.shapes_;
    if (!shapes.empty()) {
      kineto_event_.get().shapes(shapes);
      annotations_.emplace_back("Input Dims", shapesToStr(shapes));
    }

    auto& dtypes = op_event.inputs_.dtypes_;
    if (!dtypes.empty()) {
      kineto_event_.get().dtypes(dtypes);
      annotations_.emplace_back("Input type", dtypesToStr(dtypes));
    }

    if (!op_event.extra_args_.empty()) {
      kineto_event_.get().flops(
          computeFlops(op_event.name_, op_event.extra_args_));
    }
    kineto_event_.get().cuda_event_start_ =
        op_event.gpu_fallback_.cuda_event_start_;
    kineto_event_.get().cuda_event_end_ =
        op_event.gpu_fallback_.cuda_event_end_;

    // add information about an associated forward op, if a sequence number
    // is available (e.g. during training)
    if (op_event.sequence_number_ >= 0) {
      kineto_event_.get()
          .sequenceNr(op_event.sequence_number_)
          .fwdThreadId(op_event.forward_tid_);
      annotations_.emplace_back(
          "Fwd thread id", std::to_string(op_event.forward_tid_));
      annotations_.emplace_back(
          "Sequence number", std::to_string(op_event.sequence_number_));
    }
  }

  void operator()(ExtraFields<EventType::Backend>& backend_event) {
    handleJIT(backend_event);
    kineto_event_.get()
        .endThreadId(kineto_event_.get().startThreadId())
        .scope((int8_t)backend_event.scope_)
        .debugHandle(backend_event.debug_handle_)
        .backend(backend_event.backend_);

    if (!backend_event.backend_.empty()) {
      annotations_.emplace_back(
          "Backend", "\"" + backend_event.backend_ + "\"");
    }
  }

  void operator()(const ExtraFields<EventType::Allocation>& alloc) {
    kineto_event_.get()
        .deviceIndex(alloc.device_index_)
        .nBytes(alloc.alloc_size_);

    annotations_ = {
        {"Device Type", std::to_string((int8_t)alloc.device_type_)},
        {"Device Id", std::to_string(alloc.device_index_)},
        {"Addr", std::to_string(reinterpret_cast<intptr_t>(alloc.ptr_))},
        {"Bytes", std::to_string(alloc.alloc_size_)}};
    if (alloc.total_allocated_ >= 0) {
      annotations_.emplace_back(
          "Total Allocated", std::to_string(alloc.total_allocated_));
    }
    if (alloc.total_reserved_ >= 0) {
      annotations_.emplace_back(
          "Total Reserved", std::to_string(alloc.total_reserved_));
    }
  }

  template <typename T>
  void handleJIT(T& fields) {
    auto& jit_stack = fields.jit_stack_;
    auto& jit_modules = fields.jit_modules_;
    if (post_process_.get()) {
      post_process_.get()(fields.debug_handle_, jit_stack, jit_modules);
    }
    if (!jit_stack.empty()) {
      // NB: This is only for the JIT stack. The python stack (if applicable)
      //     is constructed later.
      kineto_event_.get().stack(jit_stack);
      annotations_.emplace_back(
          "Call stack", torch::profiler::impl::stacksToStr(jit_stack, ";"));
    }

    if (!jit_modules.empty()) {
      kineto_event_.get().moduleHierarchy(jit_modules);
      annotations_.emplace_back(
          "Module Hierarchy",
          torch::profiler::impl::stacksToStr(jit_modules, "."));
    }
  }

  void operator()(const ExtraFields<EventType::PyCall>& py_call) {
    addPythonAnnotations(py_call);
    if (py_call.module_.has_value()) {
      annotations_.emplace_back(
          "Python module id", std::to_string(py_call.module_->id_));
    }
  }

  void operator()(const ExtraFields<EventType::PyCCall>& py_call) {
    addPythonAnnotations(py_call);
  }

  void pushPythonMetadata(std::shared_ptr<Result> parent) {
    auto push = [&](const auto& i) {
      c10::guts::if_constexpr<std::is_base_of<
          torch::profiler::impl::PyExtraFieldsBase,
          typename std::remove_reference<decltype(i)>::type>::
                                  value>([&](auto _) {
        py_metadata_.push_back({_(i).id_, _(i).python_tid_, parent->name()});
      });
    };

    while (parent != nullptr) {
      c10::visit(push, parent->extra_fields_);
      parent = parent->parent_.lock();
    }
  }

  template <typename T>
  void addPythonAnnotations(T& t) {
    annotations_.emplace_back("Python id", std::to_string(t.id_));
    annotations_.emplace_back(
        "Python parent id",
        !py_metadata_.empty() ? std::to_string(py_metadata_.at(0).id_)
                              : "null");
    annotations_.emplace_back("Python thread", std::to_string(t.python_tid_));
  }

  void handleStack(std::weak_ptr<Result> parent) {
    // JIT stack takes precidence.
    if (!kineto_event_.get().hasStack() && !py_metadata_.empty()) {
      std::vector<std::string> stack;
      for (auto i = py_metadata_.rbegin(); i < py_metadata_.rend(); ++i) {
        stack.push_back(i->name_);
      }
      kineto_event_.get().stack(std::move(stack));
    }

    if (kineto_event_.get().hasStack()) {
      annotations_.emplace_back(
          "Call stack",
          torch::profiler::impl::stacksToStr(kineto_event_.get().stack(), ";"));
    }
  }

  struct PythonMetadata {
    size_t id_;
    size_t python_tid_;
    std::string name_;
  };

  std::reference_wrapper<KinetoEvent> kineto_event_;
  std::reference_wrapper<const post_process_t> post_process_;
  std::vector<PythonMetadata> py_metadata_;
  annotation_t annotations_;
};

// Assumption: Total threads number will not exceed 2^16-1, and total ops will
// not exceed 2^48 -1.
static inline uint64_t getForwardThreadKey(uint64_t tid, uint64_t seqNr) {
  return (((tid) << 48) | ((seqNr) & (((uint64_t)1 << 48) - 1)));
}

struct KinetoThreadLocalState : public ProfilerThreadLocalStateBase {
  explicit KinetoThreadLocalState(
      const ProfilerConfig& config,
      std::set<torch::profiler::impl::ActivityType> activities)
      : ProfilerThreadLocalStateBase(config),
        start_time_(getTimeUs()),
        record_queue_(config, activities),
        cpu_trace_(start_time_, "PyTorch Profiler") {}
  ~KinetoThreadLocalState() override = default;

  static KinetoThreadLocalState* getTLS() {
    auto tls = ProfilerThreadLocalStateBase::getTLS();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::KINETO);
    return static_cast<KinetoThreadLocalState*>(tls);
  }

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::KINETO;
  }

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      int64_t total_allocated,
      int64_t total_reserved,
      c10::Device device) override {
    if (config_.profile_memory && config_.state != ProfilerState::Disabled) {
      record_queue_.getSubqueue()->emplace_allocation_event(
          torch::profiler::impl::getApproximateTime(),
          ptr,
          alloc_size,
          total_allocated,
          total_reserved,
          device.type(),
          device.index());
    }
  }

  const post_process_t& getEventPostProcessingCallback() const {
    return event_post_process_cb_;
  }

  void setEventPostProcessingCallback(post_process_t&& cb) {
    event_post_process_cb_ = std::move(cb);
  }

  torch::profiler::impl::kineto::ActivityTraceWrapper finalizeTrace() {
    auto end_time = getTimeUs();
    record_queue_.stop();
    materializeOpEvents();

    finalizeCPUTrace(cpu_trace_.get());

    // `kineto_events_` does not include Python events. Instead it exposes them
    // via the `stacks` property.
    kineto_events_.erase(
        std::remove_if(
            kineto_events_.begin(),
            kineto_events_.end(),
            [](const auto& i) { return i.is_python_function_; }),
        kineto_events_.end());

    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      cpu_trace_.transferCpuTrace(end_time);
    }

    if (config().state != ProfilerState::KINETO_ONDEMAND) {
      auto trace = torch::profiler::impl::kineto::stopTrace();
      TORCH_CHECK(trace || !torch::profiler::kKinetoAvailable);
      addTraceEvents(trace);
      return trace;
    } else {
      return torch::profiler::impl::kineto::ActivityTraceWrapper();
    }
  }

  void materializeOpEvents() {
    std::lock_guard<std::mutex> guard(state_mutex_);
    auto converter = clock_converter_.makeConverter();

    for (auto& e : record_queue_.getRecords(converter)) {
      if (e->parent_.expired()) {
        event_tree_.push_back(e);
      }

      if (e->finished_) {
        int64_t start_us = e->start_time_ns_ / 1000;
        int64_t end_us = e->endTimeNS() / 1000;

        const auto is_python = c10::visit(
            c10::overloaded(
                [](const torch::profiler::impl::PyExtraFieldsBase&) {
                  return true;
                },
                [](const auto&) { return false; }),
            e->extra_fields_);

        kineto_events_.emplace_back(is_python);
        kineto_events_.back()
            .name(e->name())
            .startUs(start_us)
            .durationUs(end_us - start_us)
            .correlationId(e->correlationID())
            .deviceType(e->deviceType())
            .startThreadId(e->start_tid_);

        // NB: also sets fields on `kineto_events_.back()`.
        auto visitor = EventFieldsVisitor(
            e, kineto_events_.back(), getEventPostProcessingCallback());

        cpu_trace_.addCPUActivity(
            e->name(),
            e->kinetoType(),
            e->kineto_info_,
            e->correlationID(),
            start_us,
            end_us,
            visitor.annotations_);
      }
    }
  }

  void finalizeCPUTrace(
      std::unique_ptr<torch::profiler::impl::kineto::trace_t>& cpu_trace) {
#ifndef USE_KINETO
  }
#else // USE_KINETO
    TORCH_INTERNAL_ASSERT(
        cpu_trace->activities.size() == kineto_events_.size());
    // startThreadId_seqNum to pointer of activity.
    // Low-16bits of startThreadId and low-48bits seqNum are concatenated into
    // one uint64_t variable as key.

    // From the time being, we need disable the forward/backward correlation
    // feature to workaround the crash bug.
    // TODO: by Mike Guo
    // reenable the forward/backward correlation when kineto fix the following
    // raw pointer
    //    GenericTraceActivity.flow.linkedActivity
    /*
    std::unordered_map<uint64_t, libkineto::GenericTraceActivity*>
        tidSeq2activity;

    for (const auto idx : c10::irange(cpu_trace->activities.size())) {
      auto& kineto_event = kineto_events_[idx];
      auto& activity = cpu_trace->activities[idx];

      // add information about an associated forward op, if a sequence number
      // is available (e.g. during training)
      if (kineto_event.sequenceNr() >= 0) {
        generateForwardBackwardLink(
            kineto_event, fwd_bwd_link_id, activity, tidSeq2activity);
      }
    }
    */
  }

  void generateForwardBackwardLink(
      const KinetoEvent& kineto_event,
      uint64_t& fwd_bwd_link_id,
      libkineto::GenericTraceActivity& activity,
      std::unordered_map<uint64_t, libkineto::GenericTraceActivity*>&
          tidSeq2activity) {
    if (kineto_event.fwdThreadId() > 0) {
      // act is backward op.
      uint64_t key = getForwardThreadKey(
          kineto_event.fwdThreadId(), kineto_event.sequenceNr());
      auto iter = tidSeq2activity.find(key);
      if (iter != tidSeq2activity.end()) {
        libkineto::GenericTraceActivity* fwd = iter->second;
        fwd->flow.start = true;
        activity.flow.id = fwd->flow.id = fwd_bwd_link_id;
        activity.flow.type = fwd->flow.type = libkineto::kLinkFwdBwd;
        ++fwd_bwd_link_id;
      }
    } else if (kineto_event.startThreadId() != 0) {
      // act is forward op.
      uint64_t key = getForwardThreadKey(
          kineto_event.startThreadId(), kineto_event.sequenceNr());
      // Assumption: Among all ops with same sequence number,
      // the one with biggest start time is most likely launching backward op.
      auto iter = tidSeq2activity.find(key);
      if (iter == tidSeq2activity.end()) {
        tidSeq2activity[key] = &activity;
      } else {
        // Now the sequence number is only incremented on creating a "Node"
        // object for backward pass, by calling
        // "at::sequence_number::get_and_increment()". Among all ops with same
        // sequence number, the one with biggest startTime is the one launching
        // backward op.
        if (activity.startTime >= iter->second->startTime) {
          tidSeq2activity[key] = &activity;
        }
      }
    }
  }
#endif // USE_KINETO

  void addTraceEvents(
      torch::profiler::impl::kineto::ActivityTraceWrapper& trace) {
#ifdef USE_KINETO
    const auto& events = *(trace.get()->activities());
    for (const auto& ev_ptr : events) {
      if (ev_ptr == nullptr) {
        continue;
      }
      const auto& activity = *ev_ptr;
      // These events are already processed
      if (activity.type() != libkineto::ActivityType::CPU_OP &&
          activity.type() != libkineto::ActivityType::CPU_INSTANT_EVENT &&
          activity.type() != libkineto::ActivityType::USER_ANNOTATION &&
          activity.type() != libkineto::ActivityType::PYTHON_FUNCTION) {
        kineto_events_.emplace_back();
        auto& kineto_event = kineto_events_.back();
        kineto_event.name(activity.name())
            .deviceIndex(activity.deviceId())
            .deviceResourceId(activity.resourceId())
            .startUs(activity.timestamp())
            .durationUs(activity.duration())
            .activityType((uint8_t)activity.type());
        if (activity.linkedActivity()) {
          kineto_event.linkedCorrelationId(
              activity.linkedActivity()->correlationId());
        }
        kineto_event.deviceType(deviceTypeFromActivity(activity.type()));
      }
    }
#endif // USE_KINETO
  }

  uint64_t start_time_;
  torch::profiler::impl::ApproximateClockToUnixTimeConverter clock_converter_;
  torch::profiler::impl::RecordQueue record_queue_;
  torch::profiler::impl::kineto::TraceWrapper cpu_trace_;
  std::vector<KinetoEvent> kineto_events_;
  std::vector<experimental_event_t> event_tree_;
  // Optional, if event post-processing is enabled.
  post_process_t event_post_process_cb_;
};

class GlobalStateManager {
 public:
  static GlobalStateManager& singleton() {
    static GlobalStateManager singleton_;
    return singleton_;
  }

  template <typename... Args>
  static void init(Args... args) {
    if (singleton().state_) {
      LOG(WARNING) << "GlobalStatePtr already exists!";
    } else {
      singleton().state_ =
          std::make_shared<KinetoThreadLocalState>(std::forward<Args>(args)...);
    }
  }

  static auto* get() {
    return singleton().state_.get();
  }

  static std::shared_ptr<c10::DebugInfoBase> pop() {
    TORCH_INTERNAL_ASSERT(
        singleton().state_ != nullptr,
        "Global state ptr cannot be null before resetting");
    auto out = singleton().state_;
    singleton().state_.reset();
    return out;
  }

 private:
  GlobalStateManager() = default;

  std::shared_ptr<KinetoThreadLocalState> state_;
};

template <bool use_global>
static KinetoThreadLocalState* getStatePtr() {
  return c10::guts::if_constexpr<use_global>(
      [] { return GlobalStateManager::get(); },
      [] { return KinetoThreadLocalState::getTLS(); });
}

template <bool use_global_state_ptr = false>
std::unique_ptr<at::ObserverContext> onFunctionEnter(
    const at::RecordFunction& fn) {
  auto state_ptr = getStatePtr<use_global_state_ptr>();
  if (!state_ptr) {
    return nullptr;
  }
  return state_ptr->record_queue_.getSubqueue()->begin_op(fn);
}

// @lint-ignore CLANGTIDY clang-diagnostic-unused-parameter
template <bool use_global_state_ptr = false>
void onFunctionExit(
    const at::RecordFunction& fn,
    at::ObserverContext* ctx_ptr) {
  auto state_ptr = getStatePtr<use_global_state_ptr>();
  if (!state_ptr) {
    return;
  }
  const auto& config = state_ptr->config();
  auto* kineto_ctx_ptr =
      static_cast<torch::profiler::impl::KinetoObserverContext*>(ctx_ptr);
  TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);
  kineto_ctx_ptr->event_->end_time_ =
      torch::profiler::impl::getApproximateTime();
  kineto_ctx_ptr->event_->basic_fields_.end_tid_ =
      at::RecordFunction::currentThreadId();
  if (config.state == ProfilerState::KINETO_GPU_FALLBACK) {
    try {
      auto fallback = kineto_ctx_ptr->fallback_;
      TORCH_INTERNAL_ASSERT(fallback != nullptr);
      torch::profiler::impl::cudaStubs()->record(
          nullptr, &fallback->cuda_event_end_, nullptr);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to record CUDA event. " << e.what();
    }
  }

  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    torch::profiler::impl::kineto::popUserCorrelationId();
  } else {
    torch::profiler::impl::kineto::popCorrelationId();
  }
}

template <bool use_global_callback = false>
void pushProfilingCallbacks(const std::unordered_set<at::RecordScope>& scopes) {
  auto registration_state_ptr = getStatePtr<use_global_callback>();
  TORCH_INTERNAL_ASSERT(registration_state_ptr, "Expected profiler state set");
  auto recordFunctionCallback =
      at::RecordFunctionCallback(
          onFunctionEnter<use_global_callback>,
          onFunctionExit<use_global_callback>)
          .needsInputs(registration_state_ptr->config().report_input_shapes)
          .scopes(scopes);

  auto handle = c10::guts::if_constexpr<use_global_callback>(
      [&] { return at::addGlobalCallback(recordFunctionCallback); },
      [&] { return at::addThreadLocalCallback(recordFunctionCallback); });
  registration_state_ptr->setCallbackHandle(handle);
}

} // namespace

void reportBackendEventToActiveKinetoProfiler(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const at::RecordScope scope,
    const std::string& event_name,
    const std::string& backend_name) {
  TORCH_INTERNAL_ASSERT(
      GlobalStateManager::get() == nullptr,
      "On-demand profiling does not support post processing callback");

  auto state_ptr = KinetoThreadLocalState::getTLS();
  if (!state_ptr) {
    return;
  }

  state_ptr->record_queue_.getSubqueue()->emplace_backend_event(
      start_time_us,
      end_time_us,
      debug_handle,
      scope,
      event_name,
      backend_name);

  /* no support for input shapes now?
  if (config.report_input_shapes) {
    ctx_ptr->shapes = inputSizes(fn);
    ctx_ptr->dtypes = inputTypes(fn);
  }
  */
}

void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  if (config.state == ProfilerState::NVTX) {
    return;
  }
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
          config.state == ProfilerState::KINETO_GPU_FALLBACK,
      "Supported only in Kineto profiler");
  torch::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!at::hasCUDA(), activities, config.experimental_config);
}

void enableProfilerWithEventPostProcess(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    post_process_t&& cb,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      config.state != ProfilerState::NVTX,
      "NVTX does not support post processing callback.");
  TORCH_INTERNAL_ASSERT(
      GlobalStateManager::get() == nullptr,
      "On-demand profiling does not support post processing callback");

  enableProfiler(config, activities, scopes);
  auto state_ptr = KinetoThreadLocalState::getTLS();
  state_ptr->setEventPostProcessingCallback(std::move(cb));
}

void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(!profilerEnabled(), "Profiler is already enabled on this thread");
  if (config.state == ProfilerState::NVTX) {
    torch::profiler::impl::pushNVTXCallbacks(config, scopes);
    return;
  }

  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK ||
      config.state == ProfilerState::KINETO_ONDEMAND);
  TORCH_CHECK(
      !activities.empty(), "No activities specified for Kineto profiler");

  if (config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK) {
    auto state = std::make_shared<KinetoThreadLocalState>(config, activities);
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);

    if (activities.count(ActivityType::CPU)) {
      pushProfilingCallbacks<false>(scopes);
    }
    torch::profiler::impl::kineto::startTrace();
  }

  if (config.state == ProfilerState::KINETO_ONDEMAND) {
    GlobalStateManager::init(config, activities);

    TORCH_INTERNAL_ASSERT(
        activities.count(ActivityType::CPU),
        "Ondemand profiling must enable CPU tracing");
    pushProfilingCallbacks<true>(scopes);
  }
}

std::unique_ptr<ProfilerResult> disableProfiler() {
  auto state_ptr = std::static_pointer_cast<
      torch::profiler::impl::ProfilerThreadLocalStateBase>(
      GlobalStateManager::get() == nullptr
          ? c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE)
          : GlobalStateManager::pop());

  const auto& config = state_ptr->config();
  TORCH_CHECK(
      state_ptr &&
          (config.state == ProfilerState::KINETO ||
           config.state == ProfilerState::KINETO_GPU_FALLBACK ||
           config.state == ProfilerState::KINETO_ONDEMAND ||
           config.state == ProfilerState::NVTX),
      "Can't disable Kineto profiler when it's not running");

  if (state_ptr->hasCallbackHandle()) {
    at::removeCallback(state_ptr->callbackHandle());
  }

  // Traces are converged via libkineto automatically for ondemand flow
  if (state_ptr->config().state == ProfilerState::KINETO_ONDEMAND) {
    (void)std::static_pointer_cast<KinetoThreadLocalState>(state_ptr)
        ->finalizeTrace();
    return std::make_unique<ProfilerResult>();
  }

  // Shared among NVTX, KINETO, KINETO_GPU_FALLBACK
  std::unique_ptr<ProfilerResult> result;
  if (state_ptr->config().state == ProfilerState::NVTX) {
    result = std::make_unique<ProfilerResult>();
  }

  if (config.state == ProfilerState::KINETO ||
      config.state == ProfilerState::KINETO_GPU_FALLBACK) {
    auto kineto_state_ptr =
        std::static_pointer_cast<KinetoThreadLocalState>(state_ptr);
    auto trace = kineto_state_ptr->finalizeTrace();
    result = std::make_unique<ProfilerResult>(
        kineto_state_ptr->start_time_,
        std::move(kineto_state_ptr->kineto_events_),
        std::move(trace),
        std::move(kineto_state_ptr->event_tree_));
  }

  return result;
}

int64_t KinetoEvent::cudaElapsedUs() const {
  if (!cuda_event_start_ || !cuda_event_end_) {
    return -1;
  }
  try {
    return (int64_t)torch::profiler::impl::cudaStubs()->elapsed(
        &cuda_event_start_, &cuda_event_end_);
  } catch (std::exception& e) {
    LOG(WARNING) << "Failed to measure time between two CUDA events. "
                 << e.what();
  }
  return -1;
}

ProfilerResult::ProfilerResult(
    uint64_t start_time,
    std::vector<KinetoEvent> events,
    torch::profiler::impl::kineto::ActivityTraceWrapper trace,
    std::vector<experimental_event_t>&& event_tree)
    : trace_start_us_(start_time),
      events_(std::move(events)),
      trace_(std::move(trace)),
      event_tree_(std::move(event_tree)) {}
ProfilerResult::ProfilerResult() = default;
ProfilerResult::~ProfilerResult() = default;

void ProfilerResult::save(const std::string& path) {
  trace_.save(path);
}

} // namespace profiler
} // namespace autograd
} // namespace torch
