import { mount } from 'svelte'
import './app.css'
import NNTrainingView from './views/NNTrainingView.svelte'

const app = mount(NNTrainingView, {
  target: document.getElementById('app')!,
})

export default app
