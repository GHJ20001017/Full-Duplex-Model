// This is a bounded English/Chinese speech heuristic, not semantic proof.
// Quotes/code and explicitly hypothetical or negated clauses are discussions,
// not an assistant's receipt for an action taken in this turn.
function assertionClauses(value) {
  return String(value || '')
    .replace(/```[\s\S]*?```|`[^`]*`|“[^”]*”|「[^」]*」|"[^"\n]*"/g, '')
    .split(/[。！？.!?;；,，\n]/)
    .filter(clause => !/(?:\b(?:if|suppose|example|would|could|not|never|haven't|hasn't|didn't)\b|如果|假如|例如|比如|尚未|还没|没有|并未|不能|不要)/i.test(clause))
}

// Past-tense execution and concrete operation receipts are stronger than
// "started processing". This bounded vocabulary deliberately does not claim
// to establish arbitrary operation semantics from task status or result prose.
export function containsTaskCompletionClaim(value) {
  return assertionClauses(value).some(clause =>
    /(?:已经|已|刚刚|刚才).{0,16}(?:完成|执行了|执行完|处理完|重启|重装|部署|删除|安装)|(?:任务|工作|后台).{0,16}(?:完成了|完成|执行成功|处理好了)|(?:完成|执行完)(?:了)?(?:任务|工作)|(?:我|我们).{0,12}(?:执行了|完成了|重启了|部署了|删除了|安装了)|\b(?:I(?:'ve| have)?|we(?:'ve| have)?)\s+(?:(?:already|just|successfully)\s+)?(?:completed|finished|executed|restarted|rebooted|deployed|deleted|installed)\b|\b(?:task|job|work|service|server)\b.{0,35}\b(?:completed|finished|done|restarted|rebooted|deployed)\b/i.test(clause))
}

export function containsTaskStartClaim(value) {
  return assertionClauses(value).some(clause => {
    const task = /(?:后台|任务|代理|\b(?:background|task|agent|job)\b)/i.test(clause)
    if (!task) return false
    return /(?:已经|已|刚刚|刚才).{0,16}(?:提交|启动|执行|安排|派发|开始|交给|交给了)|(?:提交|启动|安排|派发|交给|执行)(?:了|成功)|(?:正在|已在).{0,12}(?:后台|执行|运行)|(?:任务|代理).{0,12}(?:已启动|已提交|开始了|运行中)|\b(?:I(?:'ve| have)?|we(?:'ve| have)?)\s+(?:(?:already|just|successfully)\s+)?(?:started|launched|submitted|dispatched|scheduled|delegated|executed|queued)\b|\b(?:task|job|agent)\b.{0,35}\b(?:has been |is |was )?(?:started|launched|submitted|dispatched|scheduled|running|queued)\b/i.test(clause)
  })
}

// Evidence must come from a server task-store lookup, not a correlation ID,
// consumesTaskNotification (which can describe failures), or a proposed call.
export function hasTaskAcceptanceEvidence(context = {}) {
  return context.taskAcceptanceVerified === true
}

export const unsupportedTaskStartGuard = Object.freeze({
  id: 'unsupported-task-start',
  instructions: '上一条回复包含未经真实任务状态和结果支持的提交、启动或完成声明，相关内容未向用户播放。不要重复该成功声明。若需要执行，请实际调用相应工具；否则说明目前可核实的状态。任务受理只代表提交，开始执行不代表已完成；只能根据真实终态及实际结果说明完成情况。',
  matches: observation => observation.origin === 'model'
    && !observation.failed
    && ((containsTaskCompletionClaim(observation.transcript)
      && observation.taskCompletionVerified !== true)
      || ((!hasTaskAcceptanceEvidence(observation)
        || (observation.taskExecutionVerified !== true
          && /启动|执行|开始|运行|\b(?:started|launched|executed|running)\b/i.test(observation.transcript)))
        && containsTaskStartClaim(observation.transcript))),
})
