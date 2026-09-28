"""Exercise actual H.264 playback, synchronization, and motion preferences.

Run a local server first, then: python tests/test_playback.py
Requires Playwright and its Chromium browser; SCENERIG_CHROME can select a
different installed Chromium executable.
"""

import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'test-results'
OUTPUT.mkdir(exist_ok=True)
BASE = os.environ.get('SCENERIG_TEST_URL', 'http://127.0.0.1:8765')
CHROME = os.environ.get('SCENERIG_CHROME') or os.environ.get('CHROMIUM_EXECUTABLE')
GROUP = '#robot-videos video'
REPLAY_HERO = '[data-hero="replay-01"]'
POLICY_HERO = '[data-hero="policy-01"]'
results = []


def check(name, fn):
    try:
        fn()
        results.append({'name': name, 'passed': True})
        print('PASS', name, flush=True)
    except Exception as error:
        results.append({'name': name, 'passed': False, 'error': str(error)})
        print('FAIL', name, str(error), flush=True)


def state(page, selector=GROUP):
    return page.locator(selector).evaluate_all('(vs) => vs.map(v => ({time:v.currentTime, paused:v.paused, ended:v.ended, duration:v.duration, ready:v.readyState, width:v.videoWidth, height:v.videoHeight, rate:v.playbackRate}))')


def wait_playing(page, selector=GROUP):
    page.wait_for_function('(s) => [...document.querySelectorAll(s)].every(v => !v.paused && v.currentTime > .2 && v.readyState >= 2)', arg=selector)


def wait_paused(page, selector=GROUP):
    page.wait_for_function('(s) => [...document.querySelectorAll(s)].every(v => v.paused)', arg=selector)


def show_hero(page, selector):
    page.locator(selector).evaluate("element => element.scrollIntoView({block: 'center'})")


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'], **({'executable_path': CHROME} if CHROME else {}))
    context = browser.new_context(viewport={'width': 1440, 'height': 1100})
    page = context.new_page()
    page.set_default_timeout(15000)
    page_errors = []
    page.on('pageerror', lambda error: page_errors.append(str(error)))
    page.goto(BASE, wait_until='networkidle')
    page.locator('#episode-select option').first.wait_for(state='attached')

    def hero_click_to_play():
        for selector in [POLICY_HERO, REPLAY_HERO]:
            show_hero(page, selector)
            page.wait_for_timeout(500)
            assert all(v['paused'] for v in state(page, f'{selector} video'))
            page.locator(f'{selector} .hero-play').click()
            wait_playing(page, f'{selector} video')
            values = state(page, f'{selector} video')
            assert all(v['rate'] == 3 for v in values), values
            if selector == REPLAY_HERO:
                assert all(12 <= v['time'] < 17 for v in values), values
        page.screenshot(path=str(OUTPUT / 'playback-hero.png'))

    check('both hero applications wait for explicit play', hero_click_to_play)

    def hero_visibility():
        page.locator('#robotics').scroll_into_view_if_needed()
        wait_paused(page, '#hero-robotics video')
        times = state(page, '#hero-robotics video')
        page.wait_for_timeout(500)
        assert all(abs(a['time'] - b['time']) < .1 for a, b in zip(times, state(page, '#hero-robotics video')))
        for selector in [POLICY_HERO, REPLAY_HERO]:
            show_hero(page, selector)
            page.wait_for_timeout(500)
            assert all(v['paused'] for v in state(page, f'{selector} video'))
            page.locator(f'{selector} .hero-play').click()
            wait_playing(page, f'{selector} video')

    check('hero pauses offscreen and needs another click to resume', hero_visibility)

    def policy_first_native_play():
        for index in range(3):
            case = context.new_page()
            try:
                case.goto(BASE, wait_until='networkidle')
                show_hero(case, POLICY_HERO)
                case.wait_for_timeout(500)
                assert all(v['paused'] for v in state(case, f'{POLICY_HERO} video'))
                video = case.locator(f'{POLICY_HERO} video').nth(index)
                box = video.bounding_box()
                video.click(position={'x': 20, 'y': box['height'] - 48})
                wait_playing(case, f'{POLICY_HERO} video')
                case.locator(f'{POLICY_HERO} .hero-play').click()
                wait_paused(case, f'{POLICY_HERO} video')
                # Only the first individual play joins the group.
                video.click(position={'x': 20, 'y': box['height'] - 48})
                case.wait_for_function('(s) => !document.querySelectorAll(s.selector)[s.index].paused', arg={'selector': f'{POLICY_HERO} video', 'index': index})
                case.wait_for_timeout(200)
                values = state(case, f'{POLICY_HERO} video')
                assert all(v['paused'] == (i != index) for i, v in enumerate(values)), values
            finally:
                case.close()

    check('first native play on any policy video starts all three; later play is independent', policy_first_native_play)

    def hero_manual_pause():
        show_hero(page, REPLAY_HERO)
        if all(v['paused'] for v in state(page, f'{REPLAY_HERO} video')):
            page.locator(f'{REPLAY_HERO} .hero-play').click()
            wait_playing(page, f'{REPLAY_HERO} video')
        page.locator(f'{REPLAY_HERO} .hero-play').click()
        wait_paused(page, f'{REPLAY_HERO} video')
        page.locator('#robotics').scroll_into_view_if_needed()
        page.wait_for_timeout(150)
        show_hero(page, REPLAY_HERO)
        page.wait_for_timeout(500)
        assert all(v['paused'] for v in state(page, f'{REPLAY_HERO} video'))

    check('manual hero pause persists after scrolling', hero_manual_pause)

    def hero_real_replay_controls():
        real = page.locator(f'{REPLAY_HERO} figure[data-role="real"] video')
        real.evaluate('(video) => video.currentTime = 8')
        page.wait_for_function('(selector) => [...document.querySelectorAll(selector)].every(video => !video.seeking && Math.abs(video.currentTime - 8) < .2)', arg=f'{REPLAY_HERO} video')
        real.evaluate('(video) => video.play()')
        wait_playing(page, f'{REPLAY_HERO} video')
        page.wait_for_timeout(500)
        times = [video['time'] for video in state(page, f'{REPLAY_HERO} video')]
        assert max(times) - min(times) < .25, times
        real.evaluate('(video) => video.pause()')
        wait_paused(page, f'{REPLAY_HERO} video')

    check('real hero replay controls synchronize all three episodes', hero_real_replay_controls)

    def hero_replay_end_restart():
        selector = f'{REPLAY_HERO} video'
        page.locator(selector).first.evaluate('(video) => video.currentTime = video.duration - .3')
        page.wait_for_function('(selector) => [...document.querySelectorAll(selector)].every(video => !video.seeking && video.duration - video.currentTime < .5)', arg=selector)
        page.locator(f'{REPLAY_HERO} .hero-play').click()
        page.wait_for_function('(selector) => [...document.querySelectorAll(selector)].every(video => video.ended)', arg=selector)
        page.locator(f'{REPLAY_HERO} .hero-play').click()
        wait_playing(page, selector)
        values = state(page, selector)
        assert all(12 <= v['time'] < 17 and v['rate'] == 3 for v in values), values
        page.locator(f'{REPLAY_HERO} .hero-play').click()
        wait_paused(page, selector)

    check('completed teaser replay restarts all three clips at 12 seconds', hero_replay_end_restart)
    page.locator('#robot-play').scroll_into_view_if_needed()

    def replay_play_pause():
        assert all(v['rate'] == 3 and v['time'] == 0 for v in state(page))
        page.locator('#robot-play').click()
        wait_playing(page)
        page.wait_for_timeout(1200)
        times = [v['time'] for v in state(page)]
        assert max(times) - min(times) < .25, times
        page.locator('#robot-play').click()
        wait_paused(page)
        assert 'Play all' in page.locator('#robot-play').inner_text()

    check('replay play-all and pause-all synchronize all three clips', replay_play_pause)

    def replay_seek():
        page.locator(GROUP).nth(1).evaluate('(v) => v.currentTime = 12')
        page.wait_for_timeout(1000)
        sought = state(page)
        assert all(abs(v['time'] - 12) < .2 for v in sought), sought
        assert all(v['paused'] for v in state(page))
        page.locator(GROUP).nth(2).evaluate('(v) => v.play()')
        wait_playing(page)
        page.locator(GROUP).nth(1).evaluate('(v) => v.pause()')
        wait_paused(page)

    check('native replay seek/play/pause controls the complete group', replay_seek)

    def replay_rate():
        page.locator(GROUP).nth(2).evaluate('(v) => v.playbackRate = 1.5')
        page.wait_for_function('() => [...document.querySelectorAll("#robot-videos video")].every(v => v.playbackRate === 1.5)')
        page.locator(GROUP).first.evaluate('(v) => v.playbackRate = 3')
        page.wait_for_function('() => [...document.querySelectorAll("#robot-videos video")].every(v => v.playbackRate === 3)')

    check('native replay playback rate synchronizes', replay_rate)

    def replay_restart():
        page.locator('#robot-restart').click()
        wait_playing(page)
        values = state(page)
        assert all(v['time'] < 2 for v in values), values
        assert max(v['time'] for v in values) - min(v['time'] for v in values) < .25, values
        page.locator('#robot-play').click()
        wait_paused(page)

    check('replay restart rewinds and plays all clips', replay_restart)

    def replay_end_restart():
        page.locator(GROUP).first.evaluate('(v) => v.currentTime = v.duration - .3')
        page.wait_for_function('() => [...document.querySelectorAll("#robot-videos video")].every(v => !v.seeking && v.duration - v.currentTime < .5)')
        page.locator('#robot-play').click()
        page.wait_for_function('() => [...document.querySelectorAll("#robot-videos video")].every(v => v.ended)')
        before = state(page)
        page.locator('#robot-play').click()
        page.wait_for_timeout(800)
        after = state(page)
        assert all(not v['paused'] and v['time'] < 6 and v['rate'] == 3 for v in after), {'before': before, 'after': after}

    check('play-all restarts a completed replay', replay_end_restart)

    def replay_rapid_pause():
        for _ in range(4):
            page.select_option('#episode-select', 'replay-02')
            page.evaluate('() => {const button=document.querySelector("#robot-play"); button.click(); button.click();}')
            page.wait_for_timeout(700)
            values = state(page)
            assert all(v['paused'] for v in values), values

    check('rapid play then pause does not restart from queued media events', replay_rapid_pause)

    def policy_independence():
        page.locator('[data-robot-kind="policy"]').click()
        page.select_option('#episode-select', 'policy-01')
        page.locator('#robot-play').click()
        wait_playing(page)
        values = state(page)
        assert all(v['rate'] == 3 for v in values), values
        assert max(v['duration'] for v in values) - min(v['duration'] for v in values) > 1, values
        page.locator('#robot-play').click()
        wait_paused(page)
        page.locator(GROUP).nth(1).evaluate('(v) => v.playbackRate = 2')
        page.wait_for_timeout(200)
        assert [v['rate'] for v in state(page)] == [3, 2, 3]
        page.locator(GROUP).nth(1).evaluate('(v) => v.playbackRate = 3')
        times = [v['time'] for v in state(page)]
        page.locator(GROUP).nth(1).evaluate('(v) => v.currentTime = 20')
        page.wait_for_timeout(400)
        after = state(page)
        assert abs(after[1]['time'] - 20) < .2
        assert abs(after[0]['time'] - times[0]) < .1 and abs(after[2]['time'] - times[2]) < .1, (times, after)
        page.locator(GROUP).nth(1).evaluate('(v) => v.play()')
        page.wait_for_timeout(500)
        after = state(page)
        assert after[0]['paused'] and not after[1]['paused'] and after[2]['paused'], after
        page.locator('#robot-play').click()
        wait_paused(page)

    check('policy videos keep independent clocks, controls, and durations', policy_independence)

    def policy_restart():
        page.locator('#robot-restart').click()
        wait_playing(page)
        assert all(v['time'] < 2 and v['rate'] == 3 for v in state(page))
        page.screenshot(path=str(OUTPUT / 'playback-policy.png'))
        page.locator('#robot-play').click()
        wait_paused(page)

    check('policy restart rewinds and starts independent clips', policy_restart)

    def episode_switch_disposes():
        page.locator('#robot-play').click()
        wait_playing(page)
        page.evaluate('window.previousVideos = [...document.querySelectorAll("#robot-videos video")]')
        page.select_option('#episode-select', 'policy-06')
        page.wait_for_timeout(300)
        assert page.evaluate('window.previousVideos.every(v => v.paused)')
        assert all(v['paused'] and v['rate'] == 3 for v in state(page))

    check('switching episodes pauses and disposes previous playback', episode_switch_disposes)

    def policy_default_rates():
        items = json.loads((ROOT / 'data/robotics.json').read_text())['items']
        for item in items:
            if item['kind'] != 'policy':
                continue
            page.select_option('#episode-select', item['id'])
            page.wait_for_function('() => { const videos = [...document.querySelectorAll("#robot-videos video")]; return videos.length === 3 && videos.every(v => v.playbackRate === 3); }')
            assert all(v['paused'] for v in state(page)), item['id']

    check('every policy episode starts at 3x after selection', policy_default_rates)

    def replay_default_rates():
        page.locator('[data-robot-kind="replay"]').click()
        items = json.loads((ROOT / 'data/robotics.json').read_text())['items']
        for item in items:
            if item['kind'] != 'replay':
                continue
            page.select_option('#episode-select', item['id'])
            page.wait_for_function('() => { const videos = [...document.querySelectorAll("#robot-videos video")]; return videos.length === 3 && videos.every(v => v.playbackRate === 3); }')
            assert all(v['paused'] and v['time'] == 0 for v in state(page)), item['id']

    check('every gallery replay episode starts at zero seconds and 3x after selection', replay_default_rates)

    def all_browser_codecs():
        files = [m for item in json.loads((ROOT / 'data/robotics.json').read_text())['items'] for m in item['media'].values()]
        records = page.evaluate('''async (files) => {
          const records = [];
          for (const file of files) {
            const video = document.createElement('video');
            video.muted = true; video.preload = 'auto'; video.src = file.src;
            document.body.append(video);
            const record = await new Promise(resolve => {
              const timeout = setTimeout(() => resolve({src:file.src, error:'metadata timeout'}), 12000);
              video.onloadeddata = () => { clearTimeout(timeout); resolve({src:file.src, width:video.videoWidth, height:video.videoHeight, duration:video.duration, playable:video.canPlayType('video/mp4; codecs="avc1.42E01E"'), error:video.error}); };
              video.onerror = () => { clearTimeout(timeout); resolve({src:file.src, error:video.error.message}); };
            });
            video.removeAttribute('src'); video.load(); video.remove(); records.push(record);
          }
          return records;
        }''', files)
        (OUTPUT / 'playback-codecs.json').write_text(json.dumps(records, indent=2) + '\n')
        assert len(records) == 48
        for record, expected in zip(records, files):
            assert record.get('width') == 640 and record.get('height') == 360 and record.get('playable') and not record['error'], record
            assert abs(record['duration'] - expected['duration']) < .15, {'actual': record, 'expected_duration': expected['duration']}

    check('all 48 H.264 videos decode in Chromium at 16:9', all_browser_codecs)

    def reduced_motion():
        reduced = browser.new_context(viewport={'width': 1440, 'height': 1100}, reduced_motion='reduce')
        reduced_page = reduced.new_page()
        reduced_page.goto(BASE, wait_until='networkidle')
        reduced_page.locator('#hero-robotics video').first.wait_for(state='attached')
        reduced_page.locator('#hero-robotics').scroll_into_view_if_needed()
        reduced_page.wait_for_timeout(800)
        assert all(v['paused'] and v['time'] == 0 for v in state(reduced_page, f'{POLICY_HERO} video'))
        reduced_page.wait_for_function('(selector) => [...document.querySelectorAll(selector)].every(video => !video.seeking && video.currentTime === 12)', arg=f'{REPLAY_HERO} video')
        assert all(v['paused'] and v['time'] == 12 for v in state(reduced_page, f'{REPLAY_HERO} video'))
        initial = reduced_page.locator('#clay-range').input_value()
        reduced_page.wait_for_timeout(500)
        assert reduced_page.locator('#clay-range').input_value() == initial
        reduced_page.locator(f'{POLICY_HERO} .hero-play').click()
        wait_playing(reduced_page, f'{POLICY_HERO} video')
        reduced.close()

    check('reduced motion disables autoplay but permits deliberate play', reduced_motion)

    def mobile_hero_visibility():
        mobile = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        mobile_page = mobile.new_page()
        mobile_page.goto(BASE, wait_until='networkidle')
        mobile_page.locator('#hero-robotics video').first.wait_for(state='attached')
        for selector in [POLICY_HERO, REPLAY_HERO]:
            show_hero(mobile_page, selector)
            mobile_page.wait_for_timeout(400)
            assert all(v['paused'] for v in state(mobile_page, f'{selector} video'))
            mobile_page.locator(f'{selector} .hero-play').click()
            wait_playing(mobile_page, f'{selector} video')
            mobile_page.locator('#robotics').scroll_into_view_if_needed()
            wait_paused(mobile_page, '#hero-robotics video')
            show_hero(mobile_page, selector)
            mobile_page.wait_for_timeout(400)
            assert all(v['paused'] for v in state(mobile_page, f'{selector} video'))
            mobile_page.locator(f'{selector} .hero-play').click()
            wait_playing(mobile_page, f'{selector} video')
        mobile_page.locator(f'{REPLAY_HERO} .hero-play').click()
        wait_paused(mobile_page, f'{REPLAY_HERO} video')
        mobile_page.locator('#robotics').scroll_into_view_if_needed()
        show_hero(mobile_page, REPLAY_HERO)
        mobile_page.wait_for_timeout(500)
        assert all(video['paused'] for video in state(mobile_page, f'{REPLAY_HERO} video'))
        mobile.close()

    check('mobile heroes require clicks and preserve pauses across visibility changes', mobile_hero_visibility)
    check('no uncaught browser errors', lambda: (_ for _ in ()).throw(AssertionError(page_errors)) if page_errors else None)
    browser.close()

(OUTPUT / 'playback-results.json').write_text(json.dumps(results, indent=2) + '\n')
assert all(result['passed'] for result in results), [result['name'] for result in results if not result['passed']]
