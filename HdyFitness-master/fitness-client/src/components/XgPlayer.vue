<template>
  <div id="XgPlayer"></div>
</template>

<script>
import Player from 'xgplayer/es/player'

export default {
  name: 'XgPlayer',
  data () {
    return {
      player: null
    }
  },
  components: {},
  mounted () {
    this.player = new Player({
      id: 'XgPlayer',
      url: 'http://localhost:8082\\video\\returnedVideo\\23023-04-21-15-31-06_sd_correct.avi'
    })
  }
}
</script>

<style scoped>

</style>
