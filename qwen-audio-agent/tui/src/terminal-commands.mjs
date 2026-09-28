const EXIT_COMMANDS = new Set(['/exit', '/quit', '/q'])

export function isExitCommand(command) {
  return EXIT_COMMANDS.has(command)
}
