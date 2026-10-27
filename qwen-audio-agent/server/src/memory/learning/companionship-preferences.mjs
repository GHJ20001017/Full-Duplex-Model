// Closed, contextual response preferences, not personality or attachment scores.
export const COMPANIONSHIP_FIELDS = Object.freeze({
  support_style: Object.freeze({
    kind: 'slot', values: Object.freeze(['listen_first', 'advice_first', 'ask_first']),
    minSpanMs: 24 * 60 * 60_000,
  }),
  humor_style: Object.freeze({
    kind: 'slot', values: Object.freeze(['no_jokes', 'gentle', 'light_teasing']),
    minSpanMs: 24 * 60 * 60_000,
  }),
  checkin_style: Object.freeze({
    kind: 'slot', values: Object.freeze(['no_unsolicited', 'occasional', 'contextual']),
    minSpanMs: 24 * 60 * 60_000,
  }),
})

export const COMPANIONSHIP_LABELS = Object.freeze({
  support_style: Object.freeze({
    listen_first: '倾诉支持：用户倾诉时先倾听和回应感受，不急于给建议',
    advice_first: '倾诉支持：用户倾诉并寻求帮助时偏好先给可执行建议',
    ask_first: '倾诉支持：用户倾诉时先问想被倾听还是一起想办法',
  }),
  humor_style: Object.freeze({
    no_jokes: '玩笑边界：不主动对用户开玩笑或调侃',
    gentle: '玩笑边界：轻松聊天时可用温和玩笑，不在严肃倾诉时逗趣',
    light_teasing: '玩笑边界：轻松聊天时可轻度打趣，不涉及敏感话题，用户不适即停止',
  }),
  checkin_style: Object.freeze({
    no_unsolicited: '会话内关心：不主动追问近况，由用户自己提起',
    occasional: '会话内关心：聊天自然涉及用户已提起的话题时偶尔关心，不连续追问',
    contextual: '会话内关心：聊天自然涉及用户已提起的话题时适度跟进，不打断任务或连续追问',
  }),
})

export const COMPANIONSHIP_GUIDE = [
  '- support_style：用户倾诉时偏好的支持方式，只能是 listen_first（先听）、advice_first（先给建议）、ask_first（先问想听还是想办法）。',
  '- humor_style：轻松聊天的玩笑边界，只能是 no_jokes（不主动开玩笑）、gentle（温和玩笑）、light_teasing（轻度打趣）。',
  '- checkin_style：正在聊天时对已提起话题的关心方式，只能是 no_unsolicited（不主动追问）、occasional（偶尔）、contextual（适度自然跟进）。不是定时或会话外联系。',
].join('\n')

export const COMPANIONSHIP_POLICY = [
  '相处偏好只用于选择回应方式，不是用户人格、心理诊断或亲密度。',
  '优先级：用户当前明确请求 > 用户明确的长期要求 > 观察推断 > 助手默认风格。',
  '先判断当前语境：严肃、难过、冲突或求助时不因玩笑偏好而打趣，不机械套用安慰模板。',
  '关心偏好仅在正在聊天且话题自然相关时使用；不凭偏好创建提醒、定时问候、会话外联系，不抢话或打断任务。',
  '不要把沉默、继续聊天、聊天时长或助手自己的话当作认可；不确定用户想倾听还是建议时自然询问。',
].join('\n')

const FIELD_TOPICS = {
  support_style: /(?:倾诉|倾听|安慰|听我(?:说|讲)|给我(?:建议|办法)|支持方式|倾诉支持|先问我.*(?:建议|办法))/,
  humor_style: /(?:玩笑|笑话|幽默|调侃|打趣|逗我|吐槽我)/,
  checkin_style: /(?:关心|问候|追问|问我近况|问我.*(?:进展|怎么样|后来)|跟进近况)/,
}
export function companionshipFieldsInText(text) {
  return Object.keys(COMPANIONSHIP_FIELDS).filter(field => FIELD_TOPICS[field].test(String(text || '')))
}

const TEMPORARY = /(?:今天|今晚|这次|本次|这回|这一回|现在|此刻|暂时|这会儿|一会儿|先别|just today|right now|this time)/i
const STANDING = /(?:以后|今后|从今|默认|每次|每回|一直|总是|不要再|别再)/
const NON_PERSONAL = /(?:台词|剧本|例句|翻译|转述|引用|角色扮演|假设|假如|如果你是|他说|她说|朋友说|同事说|妈妈说|小说|故事里|这句话|帮我写|反话|反讽|不是说|不代表|不意味着)/
const DIRECTED = /(?:你|您|请|别|不要|不用|先听我|听我说完|先问我)/
const PERSONAL = /(?:我|我们)/
const MEDICAL = /(?:抑郁症|焦虑症|病历|诊断|服药|药物|手术|自杀|自伤)/

export function isStandingCompanionshipDirective(text, field) {
  const value = String(text || '')
  return STANDING.test(value) && !TEMPORARY.test(value) && !NON_PERSONAL.test(value)
    && !MEDICAL.test(value)
    && DIRECTED.test(value)
    && companionshipFieldsInText(value).includes(field)
}

const VALUE_EVIDENCE = {
  support_style: {
    listen_first: /(?:先听我(?:说|讲)|先听我倾诉|听我说完|别急着给我建议|不要急着给我建议)/,
    advice_first: /(?:直接给我(?:建议|办法|方案)|先给我(?:建议|办法|方案))/,
    ask_first: /(?:先问我.*(?:倾听|听我|建议|办法)|问我.*(?:想要|需要).*(?:听|建议|办法))/,
  },
  humor_style: {
    no_jokes: /(?:(?:别|不要|不用).*(?:开玩笑|逗我|调侃我|打趣我)|我不喜欢你.*(?:玩笑|调侃|打趣))/,
    gentle: /(?:(?:我喜欢|我更喜欢|我能接受|我不介意).*你.*(?:温和|轻松|温柔).*玩笑|你.*(?:温和|轻松|温柔).*玩笑.*(?:我喜欢|我觉得.*(?:舒服|有趣)|我能接受))/,
    light_teasing: /(?:(?:我喜欢|我能接受|我不介意).*你.*(?:调侃我|打趣我|轻轻吐槽我)|你.*(?:调侃我|打趣我|轻轻吐槽我).*(?:我喜欢|我能接受|我觉得.*有趣))/,
  },
  checkin_style: {
    no_unsolicited: /(?:别|不要|不用).*(?:追问我|主动关心我|问候我|问我近况|问我.*(?:怎么样|进展|后来))/,
    occasional: /(?:偶尔|少一点).*(?:问我|关心我|问候我|追问我)/,
    contextual: /(?:适当.*(?:问我|关心我|问候我)|多问问我|多关心我)/,
  },
}

// Do not resolve "that was better" against generated assistant text: delivery
// is not evidence that the user heard and approved a particular response.
export function companionshipEvidenceRejection({ field, value, quote, basis = '' }, userTurns) {
  if (!COMPANIONSHIP_FIELDS[field]) return null
  if (!COMPANIONSHIP_FIELDS[field].values.includes(value)) return 'invalid_companionship_value'
  const bare = text => String(text).replace(/[^\p{L}\p{N}]+/gu, '')
  const turns = userTurns.filter(text => bare(text).includes(bare(quote)))
  if (!turns.length) return 'quote_not_from_single_user_turn'
  if (MEDICAL.test(`${quote}\n${basis}`)) return 'sensitive_companionship_context'
  const eligible = turns.filter(text => !TEMPORARY.test(text) && !NON_PERSONAL.test(text))
  if (!eligible.length) return 'temporary_or_nonpersonal_companionship'
  if (eligible.every(text => isStandingCompanionshipDirective(text, field))) {
    return 'explicit_companionship_directive'
  }
  if (!DIRECTED.test(quote) || !PERSONAL.test(quote)) return 'ambiguous_companionship_feedback'
  // Questions, negated positive preferences and comparisons need a richer
  // interpretation than this conservative first version can safely provide.
  const unambiguous = eligible.some(text => {
    if (/[?？]|(?:是不是|能不能|是否|比起)/.test(text)) return false
    if (['no_jokes', 'no_unsolicited'].includes(value)
      && /不(?:希望|想|需要|喜欢).*(?:别|不要|不用)/.test(text)) return false
    return ['no_jokes', 'no_unsolicited'].includes(value)
      || !/(?:不喜欢|不想|不需要|不希望|不要|别).*(?:先听我|先给我|直接给我|温和|轻松|打趣)/.test(text)
  })
  if (!unambiguous) return 'ambiguous_companionship_feedback'
  if (field === 'support_style' && value === 'advice_first'
    && !eligible.some(text => /(?:倾诉|倾听|安慰|情绪|心情|难过|委屈|烦恼)/.test(text))) {
    return 'support_context_missing'
  }
  if (!VALUE_EVIDENCE[field][value].test(quote)) return 'companionship_value_not_supported'
  return null
}
