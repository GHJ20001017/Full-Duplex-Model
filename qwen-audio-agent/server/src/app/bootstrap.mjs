import { createGatewayApplication } from './gateway-application.mjs'

const application = createGatewayApplication()

export const app = application.app
export const server = application.server
export const services = application.services
export const start = application.start
export const close = application.close
